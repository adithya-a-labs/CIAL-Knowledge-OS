"""Capability-aware routing for local Ollama-compatible model clients."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping


@dataclass(frozen=True, slots=True)
class ModelProfile:
    name: str
    provider: str
    model: str
    capabilities: frozenset[str] = frozenset({"text"})
    fallback_profiles: tuple[str, ...] = ()
    timeout_seconds: float = 120.0
    retries: int = 0
    temperature: float = 0.0


@dataclass(frozen=True, slots=True)
class ModelResponse:
    content: str
    model: str
    profile: str
    fallback_used: bool = False
    token_estimate: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class _OllamaLocalClient:
    """Lazy adapter for the optional local ``ollama`` Python package."""

    def generate(
        self,
        *,
        prompt: str,
        model: str,
        temperature: float = 0.0,
        timeout: float = 120.0,
        format: str | None = None,
        images: list[str] | None = None,
        **_: Any,
    ) -> Any:
        try:
            import ollama
        except ImportError as exc:
            raise RuntimeError(
                "The local ollama package is required for provider 'ollama', "
                "or inject a compatible client."
            ) from exc
        client = ollama.Client(timeout=timeout)
        options: dict[str, Any] = {"temperature": temperature}
        kwargs: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "options": options,
        }
        if format:
            kwargs["format"] = format
        if images:
            kwargs["images"] = images
        return client.generate(**kwargs)


class ModelRouter:
    """Select an offline model by agent assignment and capability tags.

    The injected client can be a callable or expose ``generate``/``invoke``.
    This keeps unit tests deterministic and avoids importing an Ollama package.
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        clients: Mapping[str, Any] | None = None,
        client_factory: Callable[[ModelProfile], Any] | None = None,
    ) -> None:
        phase5 = config.get("phase5", config)
        phase5 = phase5 if isinstance(phase5, Mapping) else {}
        raw_profiles = phase5.get("model_profiles") or {}
        self.profiles: dict[str, ModelProfile] = {}
        for name, raw in raw_profiles.items():
            if not isinstance(raw, Mapping):
                continue
            fallbacks = raw.get("fallback_profiles") or raw.get("fallback_profile") or ()
            if isinstance(fallbacks, str):
                fallbacks = (fallbacks,)
            self.profiles[str(name)] = ModelProfile(
                name=str(name),
                provider=str(raw.get("provider") or "ollama"),
                model=str(raw.get("model") or ""),
                capabilities=frozenset(
                    str(item).casefold()
                    for item in raw.get("capabilities") or ("text",)
                ),
                fallback_profiles=tuple(str(item) for item in fallbacks),
                timeout_seconds=float(raw.get("timeout_seconds") or 120),
                retries=int(raw.get("retries") or 0),
                temperature=float(raw.get("temperature") or 0),
            )
        raw_agents = phase5.get("agents") or {}
        self.agent_config = {
            str(name): dict(value) if isinstance(value, Mapping) else {}
            for name, value in raw_agents.items()
        }
        # Translate the original per-agent ``model`` shape into profiles.
        # Existing Phase 5 configurations therefore remain valid while newer
        # configurations can share capability-tagged profiles.
        for agent_name, assignment in self.agent_config.items():
            if assignment.get("model_profile") or not assignment.get("model"):
                continue
            profile_name = f"legacy_{agent_name}"
            fallback_name = f"{profile_name}_fallback"
            fallbacks: tuple[str, ...] = ()
            if assignment.get("fallback_model"):
                self.profiles[fallback_name] = ModelProfile(
                    name=fallback_name,
                    provider="ollama",
                    model=str(assignment["fallback_model"]),
                    capabilities=frozenset({"text", "structured_json"}),
                    temperature=float(assignment.get("temperature") or 0),
                )
                fallbacks = (fallback_name,)
            self.profiles[profile_name] = ModelProfile(
                name=profile_name,
                provider="ollama",
                model=str(assignment["model"]),
                capabilities=frozenset({"text", "structured_json"}),
                fallback_profiles=fallbacks,
                timeout_seconds=float(assignment.get("timeout_seconds") or 120),
                retries=int(assignment.get("retries") or 0),
                temperature=float(assignment.get("temperature") or 0),
            )
            assignment["model_profile"] = profile_name
        self.clients = dict(clients or {})
        self.client_factory = client_factory

    def profile_for(
        self,
        agent_name: str,
        *,
        required_capabilities: Iterable[str] = (),
        vision: bool = False,
    ) -> ModelProfile | None:
        assignment = self.agent_config.get(agent_name, {})
        key = "vision_model_profile" if vision else "model_profile"
        profile_name = str(assignment.get(key) or "")
        required = {str(item).casefold() for item in required_capabilities}
        if vision:
            required.add("vision")
        profile = self.profiles.get(profile_name)
        if profile and required.issubset(profile.capabilities):
            return profile
        if profile_name:
            return None
        for candidate in self.profiles.values():
            if required.issubset(candidate.capabilities):
                return candidate
        return None

    def supports(
        self, agent_name: str, capabilities: Iterable[str], *, vision: bool = False
    ) -> bool:
        return self.profile_for(
            agent_name, required_capabilities=capabilities, vision=vision
        ) is not None

    def _client(self, profile: ModelProfile) -> Any:
        client = self.clients.get(profile.name) or self.clients.get(profile.provider)
        if client is None and self.client_factory is not None:
            client = self.client_factory(profile)
        if client is None and profile.provider.casefold() == "ollama":
            client = _OllamaLocalClient()
        if client is None:
            raise RuntimeError(
                f"No local client configured for model profile '{profile.name}'."
            )
        return client

    @staticmethod
    def _invoke(client: Any, profile: ModelProfile, prompt: str, **kwargs: Any) -> str:
        options = {
            "model": profile.model,
            "temperature": profile.temperature,
            "timeout": profile.timeout_seconds,
            **kwargs,
        }
        if hasattr(client, "generate"):
            try:
                value = client.generate(prompt=prompt, **options)
            except TypeError:
                value = client.generate(prompt)
        elif hasattr(client, "invoke"):
            value = client.invoke(prompt)
        else:
            try:
                value = client(prompt=prompt, **options)
            except TypeError:
                value = client(prompt)
        if isinstance(value, Mapping):
            value = value.get("content") or value.get("response") or value.get("text")
        elif hasattr(value, "response"):
            value = value.response
        elif hasattr(value, "content"):
            value = value.content
        return str(value or "")

    def generate(
        self,
        agent_name: str,
        prompt: str,
        *,
        required_capabilities: Iterable[str] = ("text",),
        json_mode: bool = False,
        images: Iterable[str] | None = None,
    ) -> ModelResponse:
        required = set(required_capabilities)
        if json_mode:
            required.add("structured_json")
        vision = bool(images)
        profile = self.profile_for(
            agent_name, required_capabilities=required, vision=vision
        )
        if profile is None:
            raise RuntimeError(
                f"No model profile for '{agent_name}' satisfies {sorted(required)}."
            )
        candidates = (profile.name, *profile.fallback_profiles)
        errors: list[str] = []
        for candidate_index, name in enumerate(candidates):
            candidate = self.profiles.get(name)
            if candidate is None or not required.issubset(candidate.capabilities):
                continue
            for _attempt in range(candidate.retries + 1):
                try:
                    content = self._invoke(
                        self._client(candidate),
                        candidate,
                        prompt,
                        format="json" if json_mode else None,
                        images=list(images or []),
                    )
                    return ModelResponse(
                        content=content,
                        model=candidate.model,
                        profile=candidate.name,
                        fallback_used=candidate_index > 0,
                        token_estimate=max(1, len(content) // 4),
                    )
                except Exception as exc:
                    errors.append(f"{candidate.name}: {type(exc).__name__}: {exc}")
        raise RuntimeError("All local model attempts failed: " + "; ".join(errors))
