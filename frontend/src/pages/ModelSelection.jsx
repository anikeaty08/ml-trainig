import AgentConsole from "../components/AgentConsole";

export default function ModelSelection({ settings, onSettingsChange, providerCatalog }) {
  return (
    <div className="stack">
      <section className="card">
        <p className="eyebrow">Provider-first routing</p>
        <h2>Agent model selection</h2>
        <p>
          This is the agent configuration surface, not the training-model picker. The ML pipeline runs the relevant
          training families automatically; this panel only configures how the chat agent talks to providers like Ollama,
          OpenAI, Claude, Gemini, or Kimi. Settings are stored on the device through the local backend database rather
          than browser localStorage.
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
