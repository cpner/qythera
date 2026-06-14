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


# ---------------------------------------------------------------------------
# HellaSwag: Sentence Completion
# ---------------------------------------------------------------------------

class HellaSwag:
    """HellaSwag: commonsense natural language inference.

    Evaluates sentence completion with 4 candidate endings.
    Uses log-likelihood scoring to select the best completion.
    """

    def __init__(self):
        self._choice_tokens = ["A", "B", "C", "D"]

    def evaluate(
        self,
        log_likelihoods: List[List[float]],
        targets: List[int],
    ) -> dict:
        """Evaluate HellaSwag accuracy.

        Args:
            log_likelihoods: list of lists, each with 4 log-likelihoods per option
            targets: list of correct answer indices (0-3)
        """
        correct = 0
        total = len(targets)
        details = []
        for ll, target in zip(log_likelihoods, targets):
            predicted = int(np.argmax(ll))
            is_correct = predicted == target
            if is_correct:
                correct += 1
            details.append({
                "predicted": self._choice_tokens[predicted],
                "target": self._choice_tokens[target],
                "correct": is_correct,
                "log_likelihoods": ll,
            })
        return {
            "accuracy": correct / total if total > 0 else 0.0,
            "correct": correct,
            "total": total,
            "details": details,
        }

    def compute_log_likelihood(
        self,
        prompt: str,
        completions: List[str],
        token_log_probs: Callable[[str], List[float]],
    ) -> List[float]:
        """Compute sum of log-probs for each completion given the prompt."""
        results = []
        for completion in completions:
            full_text = prompt + " " + completion
            log_probs = token_log_probs(full_text)
            results.append(sum(log_probs))
        return results


# ---------------------------------------------------------------------------
# TruthfulQA
# ---------------------------------------------------------------------------

class TruthfulQA:
    """TruthfulQA: measuring truthfulness in language models.

    Supports MC1 (truthful + informative) and MC2 (truthful only) variants.
    """

    def __init__(self, variant: str = "MC1"):
        if variant not in ("MC1", "MC2"):
            raise ValueError(f"Variant must be MC1 or MC2, got {variant}")
        self.variant = variant

    def evaluate(
        self,
        log_likelihoods: List[List[float]],
        targets: List[List[int]],
    ) -> dict:
        """Evaluate TruthfulQA accuracy.

        Args:
            log_likelihoods: list of lists, each with log-likelihoods per option
            targets: list of lists of correct answer indices
        """
        correct = 0
        total = len(targets)
        details = []
        for ll, target_set in zip(log_likelihoods, targets):
            if self.variant == "MC1":
                predicted = int(np.argmax(ll))
                is_correct = predicted in target_set
            else:
                max_ll = max(ll)
                predicted = [i for i, v in enumerate(ll) if v == max_ll]
                is_correct = any(p in target_set for p in predicted)
            if is_correct:
                correct += 1
            details.append({
                "predicted": predicted,
                "targets": target_set,
                "correct": is_correct,
                "log_likelihoods": ll,
            })
        return {
            "accuracy": correct / total if total > 0 else 0.0,
            "variant": self.variant,
            "correct": correct,
            "total": total,
            "details": details,
        }


# ---------------------------------------------------------------------------
# ARC: AI2 Reasoning Challenge
# ---------------------------------------------------------------------------

class ARC:
    """ARC: science questions with multiple choice answers.

    Supports Easy and Challenging subsets.
    """

    def __init__(self, subset: str = "easy"):
        if subset not in ("easy", "challenging"):
            raise ValueError(f"Subset must be 'easy' or 'challenging', got {subset}")
        self.subset = subset
        self._choice_tokens = ["A", "B", "C", "D", "E"]

    def evaluate(
        self,
        predictions: List[str],
        targets: List[str],
    ) -> dict:
        """Evaluate ARC accuracy.

        Args:
            predictions: list of predicted answers
            targets: list of correct answers
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
            details.append({
                "predicted": pred_clean,
                "target": target_clean,
                "correct": is_correct,
            })
        return {
            "accuracy": correct / total if total > 0 else 0.0,
            "subset": self.subset,
            "correct": correct,
            "total": total,
            "details": details,
        }

    def _extract_choice(self, text: str) -> str:
        text = text.strip()
        if text and text[0].upper() in "ABCDE":
            return text[0].upper()
        m = re.search(r"\b([ABCDE])\b", text.upper())
        return m.group(1) if m else "A"


class ARC_Easy(ARC):
    """ARC Easy: science questions with multiple choice answers (easy subset)."""

    def __init__(self):
        super().__init__(subset="easy")


class ARC_Challenging(ARC):
    """ARC Challenging: science questions with multiple choice answers (challenging subset)."""

    def __init__(self):
        super().__init__(subset="challenging")


# ---------------------------------------------------------------------------
# WinoGrande: Pronoun Resolution
# ---------------------------------------------------------------------------

class WinoGrande:
    """WinoGrande: binary choice pronoun resolution benchmark."""

    def __init__(self):
        self._choice_tokens = ["A", "B"]

    def evaluate(
        self,
        predictions: List[str],
        targets: List[str],
    ) -> dict:
        """Evaluate WinoGrande accuracy.

        Args:
            predictions: list of predicted choices ('A' or 'B')
            targets: list of correct choices ('A' or 'B')
        """
        correct = 0
        total = len(targets)
        details = []
        for pred, target in zip(predictions, targets):
            pred_clean = pred.strip().upper()[:1]
            if pred_clean not in "AB":
                pred_clean = "A"
            target_clean = target.strip().upper()[:1]
            is_correct = pred_clean == target_clean
            if is_correct:
                correct += 1
            details.append({
                "predicted": pred_clean,
                "target": target_clean,
                "correct": is_correct,
            })
        return {
            "accuracy": correct / total if total > 0 else 0.0,
            "correct": correct,
            "total": total,
            "details": details,
        }


# ---------------------------------------------------------------------------
# MBPP: Mostly Basic Python Problems
# ---------------------------------------------------------------------------

class MBPP:
    """MBPP: Python code generation and evaluation benchmark."""

    def __init__(self):
        pass

    def evaluate(
        self,
        generated_code: List[str],
        test_cases: List[List[str]],
    ) -> dict:
        """Evaluate MBPP code generation.

        Args:
            generated_code: list of generated Python code strings
            test_cases: list of lists of test case strings
        """
        pass_count = 0
        total = len(generated_code)
        details = []
        for code, tests in zip(generated_code, test_cases):
            all_pass = True
            for test in tests:
                try:
                    namespace = {}
                    exec(code, namespace)
                    exec(test, namespace)
                except Exception:
                    all_pass = False
                    break
            if all_pass:
                pass_count += 1
            details.append({"passed": all_pass})
        return {
            "pass_rate": pass_count / total if total > 0 else 0.0,
            "passed": pass_count,
            "total": total,
            "details": details,
        }


# ---------------------------------------------------------------------------
# BERTScore
# ---------------------------------------------------------------------------

class BERTScore:
    """BERTScore: token embedding cosine similarity F1."""

    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim

    def _get_embeddings(self, tokens: List[str], seed: int = 0) -> np.ndarray:
        rng = np.random.RandomState(seed)
        return rng.randn(len(tokens), self.embedding_dim).astype(np.float32)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def compute(
        self,
        predictions: List[str],
        references: List[str],
    ) -> float:
        """Compute BERTScore F1 between predictions and references.

        Args:
            predictions: list of predicted strings
            references: list of reference strings

        Returns:
            Average F1 score across all pairs
        """
        result = self.evaluate(predictions, references)
        return result["bertscore_f1"]

    def evaluate(
        self,
        predictions: List[str],
        references: List[str],
    ) -> dict:
        """Evaluate BERTScore between predictions and references.

        Args:
            predictions: list of predicted strings
            references: list of reference strings
        """
        f1_scores = []
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.lower().split()
            ref_tokens = ref.lower().split()
            if not pred_tokens or not ref_tokens:
                f1_scores.append(0.0)
                continue
            pred_emb = self._get_embeddings(pred_tokens, seed=0)
            ref_emb = self._get_embeddings(ref_tokens, seed=1)
            precision_scores = []
            for i, pe in enumerate(pred_emb):
                max_sim = max(self._cosine_similarity(pe, re_emb) for re_emb in ref_emb)
                precision_scores.append(max_sim)
            recall_scores = []
            for j, re_emb in enumerate(ref_emb):
                max_sim = max(self._cosine_similarity(re_emb, pe) for pe in pred_emb)
                recall_scores.append(max_sim)
            precision = np.mean(precision_scores) if precision_scores else 0.0
            recall = np.mean(recall_scores) if recall_scores else 0.0
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0
            f1_scores.append(float(f1))
        return {
            "bertscore_f1": float(np.mean(f1_scores)) if f1_scores else 0.0,
            "per_sample_f1": f1_scores,
        }


# ---------------------------------------------------------------------------
# DiversityMetrics
# ---------------------------------------------------------------------------

class DiversityMetrics:
    """DiversityMetrics: distinct-n token diversity for response diversity."""

    def __init__(self, max_n: int = 2):
        self.max_n = max_n

    def compute(self, responses: List[str]) -> dict:
        """Compute diversity metrics.

        Args:
            responses: list of generated response strings

        Returns:
            Dictionary with distinct_1 and distinct_2 scores
        """
        result = self.evaluate(responses)
        return {
            "distinct_1": result["distinct_1"],
            "distinct_2": result["distinct_2"]
        }

    def evaluate(self, responses: List[str]) -> dict:
        """Compute distinct-1 and distinct-2 metrics.

        Args:
            responses: list of generated response strings
        """
        all_tokens = []
        all_bigrams = []
        for response in responses:
            tokens = response.lower().split()
            all_tokens.extend(tokens)
            all_bigrams.extend([(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)])
        distinct_1 = len(set(all_tokens)) / max(len(all_tokens), 1)
        distinct_2 = len(set(all_bigrams)) / max(len(all_bigrams), 1) if all_bigrams else 0.0
        return {
            "distinct_1": float(distinct_1),
            "distinct_2": float(distinct_2),
            "total_tokens": len(all_tokens),
            "total_bigrams": len(all_bigrams),
            "unique_tokens": len(set(all_tokens)),
            "unique_bigrams": len(set(all_bigrams)),
        }


# ---------------------------------------------------------------------------
# PassAtK: Unbiased Estimator
# ---------------------------------------------------------------------------

class PassAtK:
    """PassAtK: unbiased pass@k estimator via combination formula."""

    def __init__(self):
        pass

    def _combination(self, n: int, k: int) -> float:
        if k > n:
            return 0.0
        if k == 0 or k == n:
            return 1.0
        if k > n - k:
            k = n - k
        result = 1.0
        for i in range(k):
            result = result * (n - i) / (i + 1)
        return result

    def _pass_at_k(self, n: int, c: int, k: int) -> float:
        if n - c < k:
            return 1.0
        return 1.0 - self._combination(n - c, k) / self._combination(n, k)

    def compute(self, n: int, c: int, k: int) -> float:
        """Compute unbiased pass@k for a single problem.

        Args:
            n: total number of samples
            c: number of correct samples
            k: number of samples to evaluate

        Returns:
            pass@k score
        """
        return self._pass_at_k(n, c, k)

    def evaluate(
        self,
        n_total: List[int],
        n_correct: List[int],
        k: int = 1,
    ) -> dict:
        """Compute unbiased pass@k for each problem.

        Args:
            n_total: list of total samples per problem
            n_correct: list of correct samples per problem
            k: number of samples to evaluate
        """
        per_problem = []
        for n, c in zip(n_total, n_correct):
            score = self._pass_at_k(n, c, k)
            per_problem.append(score)
        avg = float(np.mean(per_problem)) if per_problem else 0.0
        return {
            "pass@k": avg,
            "k": k,
            "n_problems": len(per_problem),
            "per_problem": per_problem,
        }
