# policy.py
import torch
import torch.nn.functional as F


class Policy:
    """包装一个语言模型，提供 generate() 和 train_step_with_advantage() 两个接口。"""

    def __init__(self, model, tokenizer, lr=1e-5):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.ref_model = None  # reference model for KL penalty

    def set_ref_model(self, ref_model):
        """保存一份初始权重的拷贝，用作 KL 散度计算的锚点。"""
        self.ref_model = ref_model

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens=128) -> str:
        """推理模式：给定 prompt，生成文本。"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    @torch.no_grad()
    def get_logprobs(self, prompt: str, response: str) -> torch.Tensor:
        """计算模型对给定 response 中每个 token 的 log probability。"""
        full_text = prompt + response
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.model.device)
        logits = self.model(**inputs).logits

        prompt_len = len(self.tokenizer(prompt, return_tensors="pt")["input_ids"][0])
        response_logits = logits[:, prompt_len - 1:-1, :]
        response_ids = inputs["input_ids"][:, prompt_len:]

        logprobs = F.log_softmax(response_logits, dim=-1)
        token_logprobs = logprobs.gather(2, response_ids.unsqueeze(-1)).squeeze(-1)
        return token_logprobs

    def train_step_with_advantage(self, trajectories: list):
        """
        一个 GRPO 训练步。
        trajectories: list of (prompt, response, advantage)
        """
        losses = []
        for prompt, response, advantage in trajectories:
            new_logprobs = self.get_logprobs(prompt, response)

            if self.ref_model is not None:
                with torch.no_grad():
                    ref_logprobs = self._get_ref_logprobs(prompt, response)
                kl = (new_logprobs.exp() * (new_logprobs - ref_logprobs)).sum()
            else:
                kl = 0.0

            pg_loss = -(new_logprobs.sum() * advantage)
            loss = pg_loss + 0.1 * kl
            losses.append(loss)

        total_loss = torch.stack(losses).mean()
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.item()
