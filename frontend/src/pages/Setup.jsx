import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";

import { api } from "../services/api";

export default function Setup({ onSetupUpdate, providerCatalog, setupStatus }) {
  const navigate = useNavigate();
  const profiles = setupStatus?.profiles || [];
  const packs = setupStatus?.packs || [];
  const runtime = setupStatus?.runtime || {};

  const [profileId, setProfileId] = useState(setupStatus?.runtime?.recommended_profile || "core");
  const [selectedPackIds, setSelectedPackIds] = useState([]);
  const [selectedProviderIds, setSelectedProviderIds] = useState(["ollama"]);
  const [downloadNow, setDownloadNow] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    const nextProfileId = setupStatus?.setup_state?.selected_profile || setupStatus?.runtime?.recommended_profile || "core";
    setProfileId(nextProfileId);
    const nextPackIds = setupStatus?.setup_state?.selected_pack_ids || [];
    setSelectedPackIds(nextPackIds);
    setSelectedProviderIds(setupStatus?.setup_state?.selected_provider_ids || ["ollama"]);
    setDownloadNow(setupStatus?.setup_state?.download_now ?? true);
  }, [setupStatus]);

  useEffect(() => {
    const profile = profiles.find((item) => item.id === profileId);
    if (profile && (!selectedPackIds.length || !setupStatus?.setup_state?.selected_pack_ids?.length)) {
      setSelectedPackIds(profile.pack_ids);
    }
  }, [profileId, profiles, selectedPackIds.length, setupStatus?.setup_state?.selected_pack_ids]);

  const estimatedSize = useMemo(
    () =>
      packs
        .filter((pack) => selectedPackIds.includes(pack.id))
        .reduce((sum, pack) => sum + (pack.size_mb || 0), 0),
    [packs, selectedPackIds]
  );

  function togglePack(packId) {
    setSelectedPackIds((current) =>
      current.includes(packId) ? current.filter((item) => item !== packId) : [...current, packId]
    );
  }

  function toggleProvider(providerId) {
    setSelectedProviderIds((current) =>
      current.includes(providerId) ? current.filter((item) => item !== providerId) : [...current, providerId]
    );
  }

  async function handleSubmit(event) {
    event.preventDefault();
    setSubmitting(true);
    setError("");
    try {
      await api.initializeSetup({
        profileId,
        packIds: selectedPackIds,
        providerIds: selectedProviderIds,
        downloadNow
      });
      const nextStatus = await api.getSetupStatus();
      onSetupUpdate(nextStatus);
      navigate("/");
    } catch (submitError) {
      setError(submitError.message);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="stack">
      <section className="card hero-card">
        <div>
          <p className="eyebrow">First-run setup</p>
          <h2>Choose what to install on this device</h2>
          <p>
            Pick a setup profile, review the model packs, choose agent providers, and decide whether to download the
            selected packs now or later. This controls local capabilities only; the app itself still has no login.
          </p>
        </div>
        <div className="meta-grid">
          <div className="meta-card">
            <span>Recommended profile</span>
            <strong>{runtime.recommended_profile || "core"}</strong>
          </div>
          <div className="meta-card">
            <span>Free disk</span>
            <strong>{runtime.free_disk_gb || "?"} GB</strong>
          </div>
          <div className="meta-card">
            <span>Memory</span>
            <strong>{runtime.memory_gb || "?"} GB</strong>
          </div>
        </div>
      </section>

      <form className="stack" onSubmit={handleSubmit}>
        <section className="card">
          <h3>Install profile</h3>
          <div className="profile-grid">
            {profiles.map((profile) => (
              <label className={`select-card ${profileId === profile.id ? "selected" : ""}`} key={profile.id}>
                <input checked={profileId === profile.id} name="profile" onChange={() => {
                  setProfileId(profile.id);
                  setSelectedPackIds(profile.pack_ids);
                }} type="radio" />
                <strong>{profile.label}</strong>
                <span>{profile.description}</span>
                <span>{profile.estimated_size_mb} MB estimated</span>
              </label>
            ))}
          </div>
        </section>

        <section className="card">
          <h3>Model packs</h3>
          <div className="profile-grid">
            {packs.map((pack) => (
              <label className={`select-card ${selectedPackIds.includes(pack.id) ? "selected" : ""}`} key={pack.id}>
                <input checked={selectedPackIds.includes(pack.id)} onChange={() => togglePack(pack.id)} type="checkbox" />
                <strong>{pack.label}</strong>
                <span>{pack.description}</span>
                <span>{pack.size_mb} MB</span>
                <span>Includes: {pack.models.slice(0, 4).join(", ")}</span>
              </label>
            ))}
          </div>
        </section>

        <section className="card">
          <h3>Agent providers</h3>
          <div className="chip-row">
            {providerCatalog.map((provider) => (
              <button
                className={`button ${selectedProviderIds.includes(provider.id) ? "" : "secondary"}`}
                key={provider.id}
                onClick={(event) => {
                  event.preventDefault();
                  toggleProvider(provider.id);
                }}
                type="button"
              >
                {provider.label}
              </button>
            ))}
          </div>
          <label className="toggle-row">
            <input checked={downloadNow} onChange={(event) => setDownloadNow(event.target.checked)} type="checkbox" />
            <span>Download selected packs now</span>
          </label>
          <p>Estimated local download footprint: {estimatedSize} MB</p>
          {error ? <p className="error-text">{error}</p> : null}
          <div className="cta-row">
            <button className="button" disabled={submitting} type="submit">
              {submitting ? "Saving setup..." : "Save setup"}
            </button>
          </div>
        </section>
      </form>
    </div>
  );
}
