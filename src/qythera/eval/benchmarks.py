import math
import re
from collections import Counter
from typing import Callable, List, Optional

import numpy as np


class MMLU:
    """4-option multiple choice evaluator."""

    def evaluate(self, predictions: List[str], targets: List[str]) -> dict:
        """Evaluate MMLU multiple choice accuracy.

        Args:
            predictions: list of predicted answers ('A','B','C','D' or full text)
            targets: list of correct answers ('A','B','C','D')
        """
        correct = 0
        total = len(targets)
        details = []
        for pred, target in zip(predictions, targets):
            pred_clean = self._extract_choice(pred)
            target_clean = target.strip().upper()[:1]
            is_correct = pred_clean == target_clean
            if is_correct:
                correct += 1
            details.append({"predicted": pred_clean, "target": target_clean, "correct": is_correct})
        return {
            "accuracy": correct / total if total > 0 else 0.0,
            "correct": correct,
            "total": total,
            "details": details,
        }

    def _extract_choice(self, text: str) -> str:
        text = text.strip()
        if text and text[0].upper() in "ABCD":
            return text[0].upper()
        m = re.search(r"\b([ABCD])\b", text.upper())
        return m.group(1) if m else "A"


class HumanEval:
    """Pass@k metric for code generation."""

    def evaluate(self, generated: List[List[str]], tests: List[str], k: int = 1) -> dict:
        """Compute pass@k.

        Args:
            generated: list of lists, each inner list has k code samples
            tests: list of test strings to execute
            k: number of samples
        """
        n_samples = len(generated)
        results = []
        for samples, test_code in zip(generated, tests):
            pass_count = 0
            for sample in samples[:k]:
                try:
                    namespace = {}
                    exec(sample, namespace)
                    exec(test_code, namespace)
                    pass_count += 1
                except Exception:
                    pass
            results.append(pass_count / k)
        pass_at_k = np.mean(results) if results else 0.0
        return {
            "pass@k": float(pass_at_k),
            "k": k,
            "n_problems": n_samples,
            "per_problem": results,
        }


class GSM8K:
    """Grade school math exact match evaluator."""

    def evaluate(self, predictions: List[str], targets: List[str]) -> dict:
        """Exact match on numeric answers."""
        correct = 0
        details = []
        for pred, target in zip(predictions, targets):
            pred_num = self._extract_number(pred)
            target_num = self._extract_number(target)
            is_correct = pred_num is not None and target_num is not None and abs(pred_num - target_num) < 1e-6
            if is_correct:
                correct += 1
            details.append({
                "predicted_num": pred_num,
                "target_num": target_num,
                "correct": is_correct,
            })
        total = len(targets)
        return {
            "accuracy": correct / total if total > 0 else 0.0,
            "correct": correct,
            "total": total,
            "details": details,
        }

    def _extract_number(self, text: str) -> Optional[float]:
        text = text.replace(",", "")
        numbers = re.findall(r"[-+]?\d*\.?\d+", text)
        if numbers:
            return float(numbers[-1])
        return None


class Perplexity:
    """Compute perplexity from log probabilities."""

    def evaluate(self, log_probs: List[float]) -> dict:
        """Compute exp(mean NLL)."""
        if not log_probs:
            return {"perplexity": 0.0, "mean_nll": 0.0, "n_tokens": 0}
        mean_nll = -np.mean(log_probs)
        ppl = float(np.exp(mean_nll))
        return {
            "perplexity": ppl,
            "mean_nll": float(mean_nll),
            "n_tokens": len(log_probs),
        }


class BLEU:
    """BLEU score with n-gram precision and brevity penalty."""

    def __init__(self, max_n: int = 4):
        self.max_n = max_n

    def evaluate(self, predictions: List[str], references: List[str]) -> dict:
        bleu_scores = []
        for pred, ref in zip(predictions, references):
            bleu_scores.append(self._compute_single(pred, ref))
        return {
            "bleu": float(np.mean(bleu_scores)) if bleu_scores else 0.0,
            "per_sample": bleu_scores,
        }

    def _compute_single(self, pred: str, ref: str) -> float:
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()
        precisions = []
        for n in range(1, self.max_n + 1):
            pred_ngrams = self._ngrams(pred_tokens, n)
            ref_ngrams = self._ngrams(ref_tokens, n)
            if not pred_ngrams:
                precisions.append(0.0)
            else:
                clipped = 0
                ref_counter = Counter(ref_ngrams)
                for ng in pred_ngrams:
                    if ng in ref_counter and ref_counter[ng] > 0:
                        clipped += 1
                        ref_counter[ng] -= 1
                precisions.append(clipped / len(pred_ngrams))
        if any(p == 0 for p in precisions):
            return 0.0
        bp = min(1.0, math.exp(1 - len(ref_tokens) / max(len(pred_tokens), 1)))
        log_avg = sum(math.log(p) for p in precisions) / len(precisions)
        return bp * math.exp(log_avg)

    def _ngrams(self, tokens: list, n: int) -> list:
        return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


class ROUGE:
    """ROUGE-N and ROUGE-L evaluator."""

    def __init__(self, n: int = 2):
        self.n = n

    def evaluate(self, predictions: List[str], references: List[str]) -> dict:
        scores = {"rouge-1": [], "rouge-l": []}
        if self.n > 1:
            scores[f"rouge-{self.n}"] = []
        for pred, ref in zip(predictions, references):
            s1 = self._rouge_n(pred, ref, 1)
            scores["rouge-1"].append(s1)
            if self.n > 1:
                scores[f"rouge-{self.n}"].append(self._rouge_n(pred, ref, self.n))
            scores["rouge-l"].append(self._rouge_l(pred, ref))
        result = {}
        for key, vals in scores.items():
            result[key] = float(np.mean(vals)) if vals else 0.0
        return result

    def _rouge_n(self, pred: str, ref: str, n: int) -> float:
        pred_t = pred.lower().split()
        ref_t = ref.lower().split()
        pred_ng = Counter(self._ngrams(pred_t, n))
        ref_ng = Counter(self._ngrams(ref_t, n))
        overlap = sum((pred_ng & ref_ng).values())
        precision = overlap / sum(pred_ng.values()) if pred_ng else 0.0
        recall = overlap / sum(ref_ng.values()) if ref_ng else 0.0
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def _rouge_l(self, pred: str, ref: str) -> float:
        pred_t = pred.lower().split()
        ref_t = ref.lower().split()
        lcs_len = self._lcs_length(pred_t, ref_t)
        precision = lcs_len / len(pred_t) if pred_t else 0.0
        recall = lcs_len / len(ref_t) if ref_t else 0.0
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def _lcs_length(self, x: list, y: list) -> int:
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]

    def _ngrams(self, tokens: list, n: int) -> list:
        return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


class ECE:
    """Expected Calibration Error."""

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins

    def evaluate(self, confidences: List[float], accuracies: List[float]) -> dict:
        """Compute ECE.

        Args:
            confidences: predicted confidence per sample
            accuracies: 1 if correct, 0 if incorrect per sample
        """
        bins = np.linspace(0, 1, self.n_bins + 1)
        ece = 0.0
        bin_details = []
        n = len(confidences)
        for i in range(self.n_bins):
            mask = [(bins[i] <= c < bins[i + 1]) for c in confidences]
            if i == self.n_bins - 1:
                mask = [(bins[i] <= c <= bins[i + 1]) for c in confidences]
            in_bin = [j for j, m in enumerate(mask) if m]
            if not in_bin:
                continue
            avg_conf = np.mean([confidences[j] for j in in_bin])
            avg_acc = np.mean([accuracies[j] for j in in_bin])
            weight = len(in_bin) / n
            ece += weight * abs(avg_conf - avg_acc)
            bin_details.append({
                "bin_lower": bins[i],
                "bin_upper": bins[i + 1],
                "count": len(in_bin),
                "avg_confidence": float(avg_conf),
                "avg_accuracy": float(avg_acc),
            })
        return {
            "ece": float(ece),
            "n_bins": self.n_bins,
            "bin_details": bin_details,
        }


class ArenaELO:
    """Bradley-Terry ELO rating system."""

    def __init__(self, k: float = 32.0, initial: float = 1000.0):
        self.k = k
        self.initial = initial
        self.ratings = {}

    def _expected(self, r_a: float, r_b: float) -> float:
        return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))

    def update(self, winner: str, loser: str, draw: bool = False):
        r_w = self.ratings.get(winner, self.initial)
        r_l = self.ratings.get(loser, self.initial)
        e_w = self._expected(r_w, r_l)
        e_l = self._expected(r_l, r_w)
        if draw:
            s_w, s_l = 0.5, 0.5
        else:
            s_w, s_l = 1.0, 0.0
        self.ratings[winner] = r_w + self.k * (s_w - e_w)
        self.ratings[loser] = r_l + self.k * (s_l - e_l)

    def evaluate(self, matches: List[tuple]) -> dict:
        """Process all matches and return final ratings.

        Args:
            matches: list of (winner, loser, draw) tuples
        """
        for winner, loser, *rest in matches:
            draw = rest[0] if rest else False
            self.update(winner, loser, draw)
        sorted_ratings = sorted(self.ratings.items(), key=lambda x: -x[1])
        return {
            "ratings": dict(sorted_ratings),
            "n_matches": len(matches),
        }

    def get_rating(self, model: str) -> float:
        return self.ratings.get(model, self.initial)
