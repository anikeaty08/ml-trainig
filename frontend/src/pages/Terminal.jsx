import AgentConsole from "../components/AgentConsole";

export default function Terminal({ settings, onSettingsChange, providerCatalog, currentJobId }) {
  return (
    <div className="stack">
      <section className="card hero-card">
        <div>
          <p className="eyebrow">Terminal-first agent setup</p>
          <h2>Chat terminal and provider onboarding</h2>
          <p>
            This terminal configures only the chat agent routing. It does not pick the ML training models. The ML
            pipeline still chooses and compares the relevant training families automatically for every dataset.
          </p>
        </div>
        <div className="setup-steps">
          <strong>Use it like this</strong>
          <p>1. Choose a provider and auth mode.</p>
          <p>2. Set your main chat model ref like `openai/gpt-4.1-mini` or `ollama/llama3.2`.</p>
          <p>3. Optionally set an image model and fallback refs.</p>
          <p>4. Use `/model status`, `/model scan`, and `/onboard provider` to inspect the route.</p>
        </div>
      </section>

      <AgentConsole
        currentJobId={currentJobId}
        onSettingsChange={onSettingsChange}
        providerCatalog={providerCatalog}
        settings={settings}
      />
    </div>
  );
}
