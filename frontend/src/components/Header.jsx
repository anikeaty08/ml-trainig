export default function Header({ jobCount, activeModelRef, setupStatus }) {
  const installedPackCount = setupStatus?.packs?.filter((item) => item.installed)?.length || 0;
  const profileLabel = setupStatus?.selected_profile?.label || setupStatus?.setup_state?.selected_profile || "not set";
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
        <div className="pill">
          <span>Setup Profile</span>
          <strong>{profileLabel}</strong>
        </div>
        <div className="pill">
          <span>Installed Packs</span>
          <strong>{installedPackCount}</strong>
        </div>
        <div className="pill accent">
          <span>Agent Model</span>
          <strong>{activeModelRef || "not set"}</strong>
        </div>
      </div>
    </header>
  );
}
