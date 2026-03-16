import { useEffect, useMemo, useState } from "react";

import { api } from "../services/api";

function applyPrimaryModelRef(settings, modelRef) {
  const normalized = (modelRef || "").trim();
  if (normalized.includes("/")) {
    const [provider, model] = normalized.split("/", 2);
    return {
      ...settings,
      provider,
      model,
      primary_model_ref: normalized
    };
  }

  return {
    ...settings,
    model: normalized,
    primary_model_ref: normalized ? `${settings.provider}/${normalized}` : ""
  };
}

function parseCsvList(value) {
  return (value || "")
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function listToCsv(items) {
  return (items || []).join(", ");
}

function formatPolicy(policy) {
  if (!policy) {
    return "Policy not resolved yet.";
  }
  const chat = (policy.chat_chain || []).map((item) => item.ref).join(" -> ") || "none";
  const image = (policy.image_chain || []).map((item) => item.ref).join(" -> ") || "none";
  const authSummary = Object.entries(policy.auth_summary || {})
    .map(([provider, labels]) => `${provider}: ${labels.join(", ") || "default"}`)
    .join("\n");

  return [
    `primary: ${policy.primary_model_ref || "unset"}`,
    `image: ${policy.image_model_ref || "unset"}`,
    `fallbacks: ${(policy.fallback_model_refs || []).join(", ") || "none"}`,
    `allowlist: ${(policy.model_allowlist || []).join(", ") || "none"}`,
    `chat chain: ${chat}`,
    `image chain: ${image}`,
    `auth profiles:\n${authSummary || "none"}`
  ].join("\n");
}

function nextSettingsWithFallback(settings, action, ref) {
  const current = parseCsvList(settings.fallback_model_refs || settings.fallback_models || "");
  if (action === "add") {
    return {
      ...settings,
      fallback_model_refs: listToCsv(Array.from(new Set([...current, ref]))),
      fallback_models: listToCsv(Array.from(new Set([...current, ref])))
    };
  }
  const updated = current.filter((item) => item !== ref);
  return {
    ...settings,
    fallback_model_refs: listToCsv(updated),
    fallback_models: listToCsv(updated)
  };
}

export default function AgentConsole({ currentJobId, settings, onSettingsChange, providerCatalog }) {
  const [availableModels, setAvailableModels] = useState([]);
  const [command, setCommand] = useState("/model status");
  const [response, setResponse] = useState("Type /model status to inspect the active routing policy.");
  const [loading, setLoading] = useState(false);
  const [policy, setPolicy] = useState(null);
  const [profileForm, setProfileForm] = useState({
    label: "",
    provider: settings.provider || "ollama",
    auth_mode: "api_key",
    base_url: settings.base_url || "",
    api_key: ""
  });

  const provider = useMemo(
    () => providerCatalog.find((item) => item.id === settings.provider) || providerCatalog[0],
    [providerCatalog, settings.provider]
  );
  const suggestedModels = provider?.suggested_models || [];

  useEffect(() => {
    setProfileForm((current) => ({
      ...current,
      provider: current.provider || settings.provider || "ollama",
      base_url: current.base_url || settings.base_url || ""
    }));
  }, [settings.base_url, settings.provider]);

  async function resolvePolicy(nextSettings = settings) {
    const payload = await api.getAgentPolicy(nextSettings);
    setPolicy(payload);
    return payload;
  }

  useEffect(() => {
    if (!providerCatalog.length) {
      return;
    }
    resolvePolicy().catch(() => {});
  }, [providerCatalog.length]);

  async function refreshModels() {
    setLoading(true);
    try {
      const payload = await api.getRemoteModels(settings);
      setAvailableModels(payload.model_refs || payload.models || []);
      const resolved = await resolvePolicy(settings);
      if (payload.error) {
        setResponse(payload.error);
      } else {
        setResponse(
          `Resolved ${payload.model_refs?.length || payload.models?.length || 0} models for ${payload.normalized?.provider || settings.provider}.\n\n${formatPolicy(resolved)}`
        );
      }
    } catch (error) {
      setResponse(error.message);
    } finally {
      setLoading(false);
    }
  }

  function updateSettings(nextSettings, nextResponse) {
    onSettingsChange(nextSettings);
    resolvePolicy(nextSettings)
      .then((resolved) => setResponse(nextResponse ? `${nextResponse}\n\n${formatPolicy(resolved)}` : formatPolicy(resolved)))
      .catch((error) => setResponse(error.message));
  }

  function handleLocalModelCommand(input) {
    const parts = input.trim().split(/\s+/);
    const subcommand = parts[1];

    if (!subcommand || subcommand === "status") {
      return resolvePolicy().then((resolved) => {
        setResponse(formatPolicy(resolved));
        return true;
      });
    }

    if (subcommand === "list") {
      return resolvePolicy().then((resolved) => {
        const catalog = (resolved.model_catalog || [])
          .map((item) => `${item.alias || item.ref} -> ${item.ref} [${item.capability}]`)
          .join("\n");
        setResponse(`${formatPolicy(resolved)}\n\ncatalog:\n${catalog || "none"}`);
        return true;
      });
    }

    if (subcommand === "set" && parts[2]) {
      const nextSettings = applyPrimaryModelRef(settings, parts.slice(2).join(" "));
      updateSettings(nextSettings, `Primary model updated to ${nextSettings.primary_model_ref}`);
      return Promise.resolve(true);
    }

    if (subcommand === "image" && parts[2]) {
      const nextSettings = { ...settings, image_model_ref: parts.slice(2).join(" ") };
      updateSettings(nextSettings, `Image model updated to ${nextSettings.image_model_ref}`);
      return Promise.resolve(true);
    }

    if (subcommand === "fallback" && parts[2] && parts[3]) {
      const action = parts[2];
      const ref = parts.slice(3).join(" ");
      if (["add", "remove"].includes(action)) {
        const nextSettings = nextSettingsWithFallback(settings, action, ref);
        updateSettings(nextSettings, `Fallbacks updated: ${nextSettings.fallback_model_refs || "none"}`);
        return Promise.resolve(true);
      }
    }

    if (subcommand === "allow" && parts[2]) {
      const nextSettings = { ...settings, model_allowlist: parts.slice(2).join(" ") };
      updateSettings(nextSettings, `Allowlist updated to ${nextSettings.model_allowlist}`);
      return Promise.resolve(true);
    }

    if (subcommand === "auth" && parts[2] === "list") {
      const profiles = (settings.auth_profiles || []).map(
        (item) => `${item.label} | ${item.provider} | ${item.auth_mode} | ${item.enabled ? "enabled" : "disabled"}`
      );
      setResponse(profiles.join("\n") || "No auth profiles configured.");
      return Promise.resolve(true);
    }

    return Promise.resolve(false);
  }

  async function executeCommand(nextCommand) {
    setLoading(true);
    try {
      if (nextCommand.trim().startsWith("/model")) {
        const handled = await handleLocalModelCommand(nextCommand);
        if (handled) {
          return;
        }
      }

      const payload = await api.runConsoleCommand({
        command: nextCommand,
        settings,
        jobId: currentJobId
      });
      setResponse(payload.response);
    } catch (error) {
      setResponse(error.message);
    } finally {
      setLoading(false);
    }
  }

  async function runCommand(event) {
    event.preventDefault();
    await executeCommand(command);
  }

  function addProfile() {
    const label = profileForm.label.trim();
    if (!label) {
      setResponse("Profile label is required.");
      return;
    }

    const profile = {
      id: `${profileForm.provider}-${label}`.toLowerCase().replace(/\s+/g, "-"),
      label,
      provider: profileForm.provider,
      auth_mode: profileForm.auth_mode,
      base_url: profileForm.base_url.trim(),
      api_key: profileForm.api_key,
      enabled: true
    };

    const nextSettings = {
      ...settings,
      auth_profiles: [...(settings.auth_profiles || []).filter((item) => item.id !== profile.id), profile],
      auth_order: {
        ...(settings.auth_order || {}),
        [profile.provider]: [
          ...((settings.auth_order || {})[profile.provider] || []).filter((item) => item !== profile.id),
          profile.id
        ]
      }
    };
    updateSettings(nextSettings, `Saved auth profile ${profile.label}`);
    setProfileForm((current) => ({ ...current, label: "", api_key: "" }));
  }

  function removeProfile(profileId, providerId) {
    const nextProfiles = (settings.auth_profiles || []).filter((item) => item.id !== profileId);
    const nextOrder = {
      ...(settings.auth_order || {}),
      [providerId]: ((settings.auth_order || {})[providerId] || []).filter((item) => item !== profileId)
    };
    updateSettings(
      {
        ...settings,
        auth_profiles: nextProfiles,
        auth_order: nextOrder
      },
      `Removed auth profile ${profileId}`
    );
  }

  const chatPreview = policy?.chat_chain?.map((item) => item.ref).join(" -> ") || "none";
  const imagePreview = policy?.image_chain?.map((item) => item.ref).join(" -> ") || "none";

  return (
    <section className="card console-card">
      <div className="console-header">
        <div>
          <p className="eyebrow">OpenClaw-style agent selector</p>
          <h3>Terminal + model routing</h3>
        </div>
        <button className="button secondary" onClick={refreshModels} type="button">
          {loading ? "Checking..." : "Fetch models"}
        </button>
      </div>

      <div className="selector-grid">
        <label>
          Provider
          <select
            value={settings.provider}
            onChange={(event) =>
              updateSettings(
                {
                  ...settings,
                  provider: event.target.value,
                  base_url:
                    providerCatalog.find((item) => item.id === event.target.value)?.default_base_url || settings.base_url
                },
                `Provider switched to ${event.target.value}`
              )
            }
          >
            {providerCatalog.map((item) => (
              <option key={item.id} value={item.id}>
                {item.id}
              </option>
            ))}
          </select>
        </label>
        <label>
          Auth mode
          <select
            value={settings.auth_mode}
            onChange={(event) => updateSettings({ ...settings, auth_mode: event.target.value }, `Auth mode set to ${event.target.value}`)}
          >
            {(provider?.auth_modes || ["local"]).map((mode) => (
              <option key={mode} value={mode}>
                {mode}
              </option>
            ))}
          </select>
        </label>
        <label>
          Base URL
          <input
            value={settings.base_url}
            onChange={(event) => onSettingsChange({ ...settings, base_url: event.target.value })}
            onBlur={() => resolvePolicy().catch(() => {})}
            placeholder="http://127.0.0.1:11434"
          />
        </label>
        <label>
          Primary model ref
          <input
            value={settings.primary_model_ref || ""}
            onChange={(event) => onSettingsChange(applyPrimaryModelRef(settings, event.target.value))}
            onBlur={() => resolvePolicy(applyPrimaryModelRef(settings, settings.primary_model_ref || "")).catch(() => {})}
            placeholder="provider/model"
            list="remote-models"
          />
          <datalist id="remote-models">
            {availableModels.map((modelName) => (
              <option key={modelName} value={modelName} />
            ))}
          </datalist>
        </label>
        <label>
          Image model ref
          <input
            value={settings.image_model_ref || ""}
            onChange={(event) => onSettingsChange({ ...settings, image_model_ref: event.target.value })}
            onBlur={() => resolvePolicy().catch(() => {})}
            placeholder="provider/model"
          />
        </label>
        <label>
          Fallback model refs
          <input
            value={settings.fallback_model_refs || settings.fallback_models || ""}
            onChange={(event) =>
              onSettingsChange({
                ...settings,
                fallback_model_refs: event.target.value,
                fallback_models: event.target.value
              })
            }
            onBlur={() => resolvePolicy().catch(() => {})}
            placeholder="ollama/qwen2.5, openai/gpt-4.1-mini"
          />
        </label>
        <label>
          Allowlist
          <input
            value={settings.model_allowlist || ""}
            onChange={(event) => onSettingsChange({ ...settings, model_allowlist: event.target.value })}
            onBlur={() => resolvePolicy().catch(() => {})}
            placeholder="optional provider/model, provider/model"
          />
        </label>
        <label>
          Browser session hint
          <input
            value={settings.browser_session_hint || ""}
            onChange={(event) => onSettingsChange({ ...settings, browser_session_hint: event.target.value })}
            placeholder="Optional note for browser-login flow"
          />
        </label>
      </div>

      <section className="provider-panel">
        <div>
          <p className="eyebrow">Selected provider</p>
          <h4>{provider?.label || settings.provider}</h4>
          <p>{provider?.onboarding_hint || "Configure a local or hosted provider for chat assistance."}</p>
          <p>
            Capabilities: {(provider?.capabilities || []).join(", ") || "chat"} | Auth modes:{" "}
            {(provider?.auth_modes || []).join(", ") || "local"}
          </p>
        </div>
        <div className="chip-row">
          {suggestedModels.map((modelRef) => (
            <button className="button secondary slim-button" key={modelRef} onClick={() => updateSettings(applyPrimaryModelRef(settings, modelRef), `Primary model updated to ${modelRef}`)} type="button">
              {modelRef}
            </button>
          ))}
        </div>
      </section>

      <div className="quick-actions">
        {["/onboard", "/model status", "/model scan", "/jobs", "/dataset summary"].map((value) => (
          <button
            className="button secondary slim-button"
            key={value}
            onClick={() => {
              setCommand(value);
              executeCommand(value).catch(() => {});
            }}
            type="button"
          >
            {value}
          </button>
        ))}
      </div>

      <div className="route-preview">
        <strong>Resolved route policy</strong>
        <p>chat: {chatPreview}</p>
        <p>image: {imagePreview}</p>
      </div>

      <section className="profile-panel">
        <div className="profile-header">
          <div>
            <p className="eyebrow">Auth profiles</p>
            <h4>Provider access rotation</h4>
          </div>
        </div>
        <div className="selector-grid">
          <label>
            Label
            <input
              value={profileForm.label}
              onChange={(event) => setProfileForm({ ...profileForm, label: event.target.value })}
              placeholder="work-laptop"
            />
          </label>
          <label>
            Provider
            <select
              value={profileForm.provider}
              onChange={(event) =>
                setProfileForm({
                  ...profileForm,
                  provider: event.target.value,
                  base_url: providerCatalog.find((item) => item.id === event.target.value)?.default_base_url || profileForm.base_url
                })
              }
            >
              {providerCatalog.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.id}
                </option>
              ))}
            </select>
          </label>
          <label>
            Auth mode
            <select
              value={profileForm.auth_mode}
              onChange={(event) => setProfileForm({ ...profileForm, auth_mode: event.target.value })}
            >
              <option value="local">local</option>
              <option value="api_key">api_key</option>
              <option value="browser_login">browser_login</option>
            </select>
          </label>
          <label>
            Profile base URL
            <input
              value={profileForm.base_url}
              onChange={(event) => setProfileForm({ ...profileForm, base_url: event.target.value })}
              placeholder="http://127.0.0.1:11434"
            />
          </label>
          <label>
            Profile API key
            <input
              type="password"
              value={profileForm.api_key}
              onChange={(event) => setProfileForm({ ...profileForm, api_key: event.target.value })}
              placeholder="Optional"
            />
          </label>
        </div>
        <div className="cta-row">
          <button className="button secondary" onClick={addProfile} type="button">
            Save profile
          </button>
        </div>
        <div className="profile-list">
          {(settings.auth_profiles || []).length === 0 ? <p>No auth profiles yet. The default route still works.</p> : null}
          {(settings.auth_profiles || []).map((profileItem) => (
            <article className="job-row" key={profileItem.id}>
              <strong>{profileItem.label}</strong>
              <span>
                {profileItem.provider} | {profileItem.auth_mode}
              </span>
              <span>{profileItem.base_url || "default base URL"}</span>
              <button
                className="button secondary slim-button"
                onClick={() => removeProfile(profileItem.id, profileItem.provider)}
                type="button"
              >
                Remove
              </button>
            </article>
          ))}
        </div>
      </section>

      <form className="terminal-shell" onSubmit={runCommand}>
        <div className="terminal-output">{response}</div>
        <div className="terminal-input">
          <span>$</span>
          <input value={command} onChange={(event) => setCommand(event.target.value)} />
          <button className="button" type="submit">
            Run
          </button>
        </div>
      </form>
    </section>
  );
}
