import AgentConsole from "../components/AgentConsole";

export default function ModelSelection({ settings, onSettingsChange, providerCatalog }) {
  return (
    <div className="stack">
      <section className="card">
        <p className="eyebrow">Provider-first routing</p>
        <h2>Agent model selection</h2>
        <p>
          This follows the OpenClaw-style pattern: choose a provider first, then a primary model, then optional
          fallback models. Settings are saved in browser localStorage on this machine.
        </p>
      </section>

      <AgentConsole
        currentJobId={null}
        settings={settings}
        onSettingsChange={onSettingsChange}
        providerCatalog={providerCatalog}
      />
    </div>
  );
}
