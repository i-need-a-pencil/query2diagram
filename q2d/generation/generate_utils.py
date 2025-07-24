import asyncio
from abc import ABC
from collections.abc import Awaitable, Callable
from time import sleep
from typing import Any, Dict, List, Optional, Type

from openai import AsyncOpenAI, OpenAI
from q2d.common.types import GeneratedOutput, SubscriptableBaseModel
from q2d.generation.generate_api import apply_template
from tqdm.asyncio import tqdm_asyncio
from tqdm.auto import tqdm

GEN_FUNC_TYPE = Callable[[List[Dict[str, str]]], GeneratedOutput]
GEN_FUNC_TYPE_ASYNC = Callable[[List[Dict[str, str]]], Awaitable[GeneratedOutput]]


class InferenceEngineConfig(ABC, SubscriptableBaseModel): ...


class OpenAIEngineConfig(InferenceEngineConfig):
    model_url: str
    api_key: str
    response_format: Optional[Any] = None
    continue_messages: Optional[List[Dict[str, str]]] = None
    continue_response_format: Optional[Type[SubscriptableBaseModel]] = None
    retries_num: int = 0
    retry_after: int = 0
    extraction_function: Optional[Callable[[Optional[str]], Optional[str]]] = None
    extraction_raise_exception: bool = False
    stream: bool = False
    model_params: Dict[str, Any]
    client_timeout: int = 3600 * 24


class InferenceEngine(ABC):
    def __init__(self, config: InferenceEngineConfig): ...

    def generate_batched(self, messages_batch: List[List[Dict[str, str]]]) -> List[GeneratedOutput]:
        raise NotImplementedError("InferenceEngine.generate_batched is not implemented")

    def generate_single(self, messages: List[Dict[str, str]]) -> GeneratedOutput:
        return self.generate_batched(messages_batch=[messages])[0]


class OpenAIEngine(InferenceEngine):
    def __init__(self, config: InferenceEngineConfig) -> None:
        assert type(config) == OpenAIEngineConfig, "Wrong config type"

        client = OpenAI(base_url=config.model_url, api_key=config.api_key, max_retries=1, timeout=config.client_timeout)
        if config.stream:
            self.generation_function = self.get_openai_streaming_generator(
                client=client, response_format=config.response_format, **config.model_params
            )
        else:
            self.generation_function = self.get_openai_generator(
                client=client, response_format=config.response_format, **config.model_params
            )

        if config.continue_messages is not None:
            if config.stream:
                continue_generation_function = self.get_openai_streaming_generator(
                    client=client, response_format=config.continue_response_format, **config.model_params
                )
            else:
                continue_generation_function = self.get_openai_generator(
                    client=client, response_format=config.continue_response_format, **config.model_params
                )
            self.generation_function = self.continue_chain_wrapper(
                self.generation_function, continue_generation_function, config.continue_messages
            )

        if config.extraction_function is not None:
            self.generation_function = self.extractor_wrapper(
                self.generation_function,
                extraction_function=config.extraction_function,
                raise_exception=config.extraction_raise_exception,
            )

        if config.retries_num >= 0:
            self.generation_function = self.retry_wrapper(
                self.generation_function, retries_num=config.retries_num, retry_after=config.retry_after
            )

    def retry_wrapper(
        self, generation_function: GEN_FUNC_TYPE, retries_num: int = 3, retry_after: int = 0
    ) -> GEN_FUNC_TYPE:
        def wrapper(messages: List[Dict[str, str]]) -> GeneratedOutput:
            retries = 0
            while retries <= retries_num:
                try:
                    return generation_function(messages)
                except Exception as e:
                    print(str(e), f"; retry number {retries}")

                    if retry_after > 0:
                        sleep(retry_after)

                retries += 1
            return GeneratedOutput()

        return wrapper

    def continue_chain_wrapper(
        self,
        generation_function_first: GEN_FUNC_TYPE,
        generation_function_second: GEN_FUNC_TYPE,
        continue_messages: List[Dict[str, str]],
    ) -> GEN_FUNC_TYPE:
        def wrapper(messages: List[Dict[str, str]]) -> GeneratedOutput:
            generated_output_first = generation_function_first(messages)
            if not generated_output_first.is_empty():
                messages += apply_template(None, None, {}, generated_output_first.raw_output)
            messages += continue_messages
            generated_output_second = generation_function_second(messages)
            return generated_output_second

        return wrapper

    def extractor_wrapper(
        self,
        generation_function: GEN_FUNC_TYPE,
        extraction_function: Callable[[Optional[str]], Optional[str]],
        raise_exception: bool,
    ) -> GEN_FUNC_TYPE:
        def wrapper(messages: List[Dict[str, str]]) -> GeneratedOutput:
            generated_output = generation_function(messages)
            generated_output["extracted_output"] = extraction_function(generated_output["raw_output"])
            if raise_exception and generated_output["extracted_output"] is None:
                raise ValueError("Output can not be extracted")
            return generated_output

        return wrapper

    def get_openai_generator(self, client: Any, response_format: Optional[Any], **model_params: Any) -> GEN_FUNC_TYPE:
        def generator(messages: List[Dict[str, str]]) -> GeneratedOutput:
            if response_format is None:
                completion = client.chat.completions.create(
                    messages=messages,
                    **model_params,
                )
            else:
                completion = client.beta.chat.completions.parse(
                    messages=messages,
                    response_format=response_format,
                    **model_params,
                )
            sleep(1)
            if type(completion) == str:
                raise ConnectionRefusedError(f"Wrong output have been received in request: {completion[:50]}...")
            return GeneratedOutput(raw_output=completion.choices[0].message.content)

        return generator

    def get_openai_streaming_generator(
        self, client: Any, response_format: Optional[Any], **model_params: Any
    ) -> GEN_FUNC_TYPE:
        def generator(messages: List[Dict[str, str]]) -> GeneratedOutput:
            content = ""
            total_tokens = 0

            if response_format is None:
                completion = client.chat.completions.create(
                    messages=messages,
                    stream=True,
                    stream_options={"include_usage": True},
                    **model_params,
                )

                # to avoid error responses
                if type(completion) == str:
                    raise ConnectionRefusedError(f"Wrong output have been received in request: {completion[:50]}...")

                for chunk in completion:
                    content += chunk.choices[0].delta.content if len(chunk.choices) > 0 else ""
                    total_tokens = chunk.usage.total_tokens if chunk.usage is not None else 0
            else:
                with client.beta.chat.completions.stream(
                    messages=messages,
                    response_format=response_format,
                    stream_options={"include_usage": True},
                    **model_params,
                ) as stream:
                    for stream_event in stream:
                        ...
                final_completion = stream.get_final_completion()
                content = final_completion.choices[0].message.content if len(final_completion.choices) > 0 else ""
                total_tokens = final_completion.usage.total_tokens if final_completion.usage is not None else 0
            return GeneratedOutput(raw_output=content, total_tokens=total_tokens)

        return generator

    def generate_batched(self, messages_batch: List[List[Dict[str, str]]]) -> List[GeneratedOutput]:
        return [self.generation_function(message) for message in tqdm(messages_batch, desc="OpenAIEngine Generation")]


class AsyncOpenAIEngine(InferenceEngine):
    def __init__(self, config: InferenceEngineConfig) -> None:
        assert type(config) == OpenAIEngineConfig, "Wrong config type"

        client = AsyncOpenAI(
            base_url=config.model_url, api_key=config.api_key, max_retries=1, timeout=config.client_timeout
        )
        if config.stream:
            self.generation_function = self.get_openai_streaming_generator(
                client=client, response_format=config.response_format, **config.model_params
            )
        else:
            self.generation_function = self.get_openai_generator(
                client=client, response_format=config.response_format, **config.model_params
            )

        if config.continue_messages is not None:
            if config.stream:
                continue_generation_function = self.get_openai_streaming_generator(
                    client=client, response_format=config.continue_response_format, **config.model_params
                )
            else:
                continue_generation_function = self.get_openai_generator(
                    client=client, response_format=config.continue_response_format, **config.model_params
                )
            self.generation_function = self.continue_chain_wrapper(
                self.generation_function, continue_generation_function, config.continue_messages
            )

        if config.extraction_function is not None:
            self.generation_function = self.extractor_wrapper(
                self.generation_function,
                extraction_function=config.extraction_function,
                raise_exception=config.extraction_raise_exception,
            )
        if config.retries_num >= 0:
            self.generation_function = self.retry_wrapper(
                self.generation_function, retries_num=config.retries_num, retry_after=config.retry_after
            )

    def retry_wrapper(
        self, generation_func: GEN_FUNC_TYPE_ASYNC, retries_num: int = 3, retry_after: int = 0
    ) -> GEN_FUNC_TYPE_ASYNC:
        LOCKER = asyncio.Lock()

        async def wrapper(messages: List[Dict[str, str]]) -> GeneratedOutput:
            retries = 0
            while retries <= retries_num:
                try:
                    return await generation_func(messages)
                except Exception as e:
                    print(str(e), f"; retry number {retries}")
                    async with LOCKER:
                        if retry_after > 0:
                            await asyncio.sleep(retry_after)
                retries += 1
            return GeneratedOutput()

        return wrapper

    def continue_chain_wrapper(
        self,
        generation_function_first: GEN_FUNC_TYPE_ASYNC,
        generation_function_second: GEN_FUNC_TYPE_ASYNC,
        continue_messages: List[Dict[str, str]],
    ) -> GEN_FUNC_TYPE_ASYNC:
        async def wrapper(messages: List[Dict[str, str]]) -> GeneratedOutput:
            generated_output_first = await generation_function_first(messages)
            if not generated_output_first.is_empty():
                messages += apply_template(None, None, {}, generated_output_first.raw_output)
            messages += continue_messages
            generated_output_second = await generation_function_second(messages)
            return generated_output_second

        return wrapper

    def extractor_wrapper(
        self,
        generation_function: GEN_FUNC_TYPE_ASYNC,
        extraction_function: Callable[[Optional[str]], Optional[str]],
        raise_exception: bool,
    ) -> GEN_FUNC_TYPE_ASYNC:
        async def wrapper(messages: List[Dict[str, str]]) -> GeneratedOutput:
            generated_output = await generation_function(messages)
            generated_output["extracted_output"] = extraction_function(generated_output["raw_output"])
            if raise_exception and generated_output["extracted_output"] is None:
                raise ValueError("Output can not be extracted")
            return generated_output

        return wrapper

    def get_openai_generator(
        self, client: Any, response_format: Optional[Any], **model_params: Any
    ) -> GEN_FUNC_TYPE_ASYNC:
        async def generator(messages: List[Dict[str, str]]) -> GeneratedOutput:
            if response_format is None:
                completion = await client.chat.completions.create(
                    messages=messages,
                    **model_params,
                )
            else:
                completion = await client.beta.chat.completions.parse(
                    messages=messages,
                    response_format=response_format,
                    **model_params,
                )

            # todo: should make configurable later
            sleep(1)
            if type(completion) == str:
                raise ConnectionRefusedError(f"Wrong output have been received in request: {completion[:50]}...")
            return GeneratedOutput(raw_output=completion.choices[0].message.content)

        return generator

    def get_openai_streaming_generator(
        self, client: Any, response_format: Optional[Any], **model_params: Any
    ) -> GEN_FUNC_TYPE_ASYNC:
        async def generator(messages: List[Dict[str, str]]) -> GeneratedOutput:
            content = ""
            total_tokens = 0

            if response_format is None:
                stream = await client.chat.completions.create(
                    messages=messages,
                    stream=True,
                    stream_options={"include_usage": True},
                    **model_params,
                )

                async for chunk in stream:
                    if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta.content:
                        content += chunk.choices[0].delta.content
                    if chunk.usage is not None:
                        total_tokens = chunk.usage.total_tokens
            else:
                async with client.beta.chat.completions.stream(
                    messages=messages,
                    response_format=response_format,
                    stream_options={"include_usage": True},
                    **model_params,
                ) as stream:
                    async for chunk in stream:
                        ...
                    final_completion = await stream.get_final_completion()
                    content = final_completion.choices[0].message.content if len(final_completion.choices) > 0 else ""
                    total_tokens = final_completion.usage.total_tokens if final_completion.usage is not None else 0
            return GeneratedOutput(raw_output=content, total_tokens=total_tokens)

        return generator

    def generate_batched(self, messages_batch: List[List[Dict[str, str]]]) -> List[GeneratedOutput]:
        generated_output_tasks = [self.generation_function(messages) for messages in messages_batch]
        generated_output = asyncio.run(
            tqdm_asyncio.gather(*generated_output_tasks, desc="AsyncOpenAIEngine Generation")
        )
        return generated_output
