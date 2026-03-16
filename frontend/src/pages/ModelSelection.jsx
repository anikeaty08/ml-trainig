import AgentConsole from "../components/AgentConsole";

export default function ModelSelection({ settings, onSettingsChange, providerCatalog }) {
  return (
    <div className="stack">
      <section className="card">
        <p className="eyebrow">Provider-first routing</p>
        <h2>Agent model configuration</h2>
        <p>
          This page is only for the chat agent. It tells the terminal which provider and `provider/model` ref to use
          when you ask questions or run agent commands. It does not choose the ML training models. Training-model
          selection stays automatic for every dataset.
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
