export default function Header({ jobCount, activeModelRef }) {
  return (
    <header className="app-header">
      <div>
        <p className="eyebrow">Local-First Autonomous ML</p>
        <h1>ML Pipeline Agent</h1>
      </div>
      <div className="header-stats">
        <div className="pill">
          <span>Jobs</span>
          <strong>{jobCount}</strong>
        </div>
        <div className="pill accent">
          <span>Agent Model</span>
          <strong>{activeModelRef || "not set"}</strong>
        </div>
      </div>
    </header>
  );
}
