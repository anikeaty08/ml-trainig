import { useMemo, useState } from "react";

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

export default function AgentConsole({ currentJobId, settings, onSettingsChange, providerCatalog }) {
  const [availableModels, setAvailableModels] = useState([]);
  const [command, setCommand] = useState("help");
  const [response, setResponse] = useState("Type help to see available commands.");
  const [loading, setLoading] = useState(false);

  const provider = useMemo(
    () => providerCatalog.find((item) => item.id === settings.provider) || providerCatalog[0],
    [providerCatalog, settings.provider]
  );

  async function refreshModels() {
    setLoading(true);
    try {
      const payload = await api.getRemoteModels(settings);
      setAvailableModels(payload.model_refs || payload.models || []);
      if (payload.error) {
        setResponse(payload.error);
      } else if ((payload.model_refs || []).length) {
        setResponse(`Resolved ${payload.model_refs.length} models for ${payload.normalized?.provider || settings.provider}.`);
      }
    } catch (error) {
      setResponse(error.message);
    } finally {
      setLoading(false);
    }
  }

  async function runCommand(event) {
    event.preventDefault();
    setLoading(true);
    try {
      const payload = await api.runConsoleCommand({
        command,
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
              onSettingsChange({
                ...settings,
                provider: event.target.value,
                base_url:
                  providerCatalog.find((item) => item.id === event.target.value)?.default_base_url || settings.base_url,
                primary_model_ref: settings.primary_model_ref
                  ? settings.primary_model_ref.replace(/^[^/]+/, event.target.value)
                  : settings.primary_model_ref
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
            value={settings.auth_mode}
            onChange={(event) => onSettingsChange({ ...settings, auth_mode: event.target.value })}
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
            placeholder="http://127.0.0.1:11434"
          />
        </label>
        <label>
          Primary model ref
          <input
            value={settings.primary_model_ref || ""}
            onChange={(event) => onSettingsChange(applyPrimaryModelRef(settings, event.target.value))}
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
            placeholder="ollama/qwen2.5, openai/gpt-4.1-mini"
          />
        </label>
        <label>
          API key
          <input
            type="password"
            value={settings.api_key}
            onChange={(event) => onSettingsChange({ ...settings, api_key: event.target.value })}
            placeholder="Optional"
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

      <div className="route-preview">
        <strong>Route chain</strong>
        <p>
          {[settings.primary_model_ref, ...(settings.fallback_model_refs || "").split(",").map((item) => item.trim())]
            .filter(Boolean)
            .join(" -> ") || `${settings.provider}/<choose-model>`}
        </p>
      </div>

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
