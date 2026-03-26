import requests
from typing import Any, List

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class MyLocalChatModel(BaseChatModel):
    api_url: str = "xxx/chat"

    @property
    def _llm_type(self) -> str:
        return "my_local_qwen_lora"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: List[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any
    ) -> ChatResult:
        payload_messages = []

        for m in messages:
            if isinstance(m, HumanMessage):
                role = "user"
            elif isinstance(m, AIMessage):
                role = "assistant"
            elif isinstance(m, SystemMessage):
                role = "system"
            else:
                role = "user"

            payload_messages.append({
                "role": role,
                "content": m.content
            })

        response = requests.post(
            self.api_url,
            json={"messages": payload_messages},
            timeout=120
        )
        response.raise_for_status()

        content = response.json()["content"]
        message = AIMessage(content=content)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])