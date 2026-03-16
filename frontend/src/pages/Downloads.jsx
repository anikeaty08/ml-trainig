import { useMemo, useState } from "react";

import { api } from "../services/api";

export default function Downloads({ onSetupUpdate, setupStatus }) {
  const packs = setupStatus?.packs || [];
  const runtime = setupStatus?.runtime || {};
  const [workingPackId, setWorkingPackId] = useState("");
  const [error, setError] = useState("");

  const totals = useMemo(() => {
    const installed = packs.filter((pack) => pack.installed);
    return {
      count: installed.length,
      sizeMb: installed.reduce((sum, pack) => sum + (pack.size_mb || 0), 0)
    };
  }, [packs]);

  async function refresh() {
    const nextStatus = await api.getSetupStatus();
    onSetupUpdate(nextStatus);
  }

  async function handleInstall(packId) {
    setWorkingPackId(packId);
    setError("");
    try {
      await api.installPacks([packId]);
      await refresh();
    } catch (nextError) {
      setError(nextError.message);
    } finally {
      setWorkingPackId("");
    }
  }

  async function handleRemove(packId) {
    const confirmed = window.confirm(`Remove local pack ${packId}?`);
    if (!confirmed) {
      return;
    }
    setWorkingPackId(packId);
    setError("");
    try {
      await api.removePack(packId);
      await refresh();
    } catch (nextError) {
      setError(nextError.message);
    } finally {
      setWorkingPackId("");
    }
  }

  return (
    <div className="stack">
      <section className="card hero-card">
        <div>
          <p className="eyebrow">Download manager</p>
          <h2>Local packs and storage</h2>
          <p>
            Install or remove local capability packs, review what is already present on disk, and keep track of the
            resource footprint on this machine.
          </p>
        </div>
        <div className="meta-grid">
          <div className="meta-card">
            <span>Installed packs</span>
            <strong>{totals.count}</strong>
          </div>
          <div className="meta-card">
            <span>Approx pack size</span>
            <strong>{totals.sizeMb} MB</strong>
          </div>
          <div className="meta-card">
            <span>Free disk</span>
            <strong>{runtime.free_disk_gb || "?"} GB</strong>
          </div>
        </div>
      </section>

      <section className="card">
        <h3>Pack inventory</h3>
        {error ? <p className="error-text">{error}</p> : null}
        <div className="profile-grid">
          {packs.map((pack) => (
            <article className={`select-card ${pack.installed ? "selected" : ""}`} key={pack.id}>
              <strong>{pack.label}</strong>
              <span>{pack.description}</span>
              <span>{pack.size_mb} MB</span>
              <span>Models: {pack.models.join(", ")}</span>
              <div className="cta-row">
                {pack.installed ? (
                  <button className="button secondary slim-button" disabled={workingPackId === pack.id} onClick={() => handleRemove(pack.id)} type="button">
                    {workingPackId === pack.id ? "Removing..." : "Remove"}
                  </button>
                ) : (
                  <button className="button slim-button" disabled={workingPackId === pack.id} onClick={() => handleInstall(pack.id)} type="button">
                    {workingPackId === pack.id ? "Installing..." : "Install"}
                  </button>
                )}
              </div>
            </article>
          ))}
        </div>
      </section>
    </div>
  );
}
