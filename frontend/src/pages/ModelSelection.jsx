import AgentConsole from "../components/AgentConsole";

export default function ModelSelection({ settings, onSettingsChange, providerCatalog }) {
  return (
    <div className="stack">
      <section className="card">
        <p className="eyebrow">Provider-first routing</p>
        <h2>Agent model selection</h2>
        <p>
          This follows the OpenClaw-style pattern more closely now: primary model, image model, fallback chain,
          allowlist, and provider auth profiles. Settings are saved in browser localStorage on this machine and passed
          to the backend only when needed for commands or runs.
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
