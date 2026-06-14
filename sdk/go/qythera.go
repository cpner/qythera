package qythera

import (
    "bytes"
    "encoding/json"
    "fmt"
    "net/http"
    "time"
)

type Client struct {
    BaseURL string
    HTTP    *http.Client
    APIKey  string
}

type Message struct {
    Role    string `json:"role"`
    Content string `json:"content"`
}

type ChatRequest struct {
    Messages    []Message `json:"messages"`
    Model       string    `json:"model,omitempty"`
    Temperature float64   `json:"temperature,omitempty"`
    MaxTokens   int       `json:"max_tokens,omitempty"`
}

type ChatResponse struct {
    Choices []struct {
        Message Message `json:"message"`
    } `json:"choices"`
}

func NewClient(baseURL, apiKey string) *Client {
    return &Client{BaseURL: baseURL, HTTP: &http.Client{Timeout: 120 * time.Second}, APIKey: apiKey}
}

func (c *Client) Chat(messages []Message) (*ChatResponse, error) {
    req := ChatRequest{Messages: messages, Temperature: 0.7, MaxTokens: 2048}
    body, _ := json.Marshal(req)
    httpReq, _ := http.NewRequest("POST", c.BaseURL+"/v1/chat/completions", bytes.NewReader(body))
    httpReq.Header.Set("Content-Type", "application/json")
    if c.APIKey != "" { httpReq.Header.Set("Authorization", "Bearer "+c.APIKey) }
    resp, err := c.HTTP.Do(httpReq)
    if err != nil { return nil, err }
    defer resp.Body.Close()
    var result ChatResponse
    json.NewDecoder(resp.Body).Decode(&result)
    return &result, nil
}

func (c *Client) Generate(prompt string) (string, error) {
    msgs := []Message{{Role: "user", Content: prompt}}
    resp, err := c.Chat(msgs)
    if err != nil { return "", err }
    if len(resp.Choices) == 0 { return "", fmt.Errorf("no response") }
    return resp.Choices[0].Message.Content, nil
}

func (c *Client) Health() bool {
    resp, err := http.Get(c.BaseURL + "/health")
    return err == nil && resp.StatusCode == 200
}
