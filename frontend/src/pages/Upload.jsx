import { useState } from "react";
import { useNavigate } from "react-router-dom";

import { api } from "../services/api";

export default function Upload({ settings }) {
  const navigate = useNavigate();
  const [file, setFile] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");

  async function handleSubmit(event) {
    event.preventDefault();
    if (!file) {
      setError("Choose a CSV or TSV file first.");
      return;
    }

    setSubmitting(true);
    setError("");
    try {
      const payload = await api.submitJob({ file, settings });
      navigate(`/processing/${payload.job_id}`);
    } catch (submitError) {
      setError(submitError.message);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="stack">
      <section className="card">
        <p className="eyebrow">Drop a dataset and go</p>
        <h2>Upload data</h2>
        <p>
          The pipeline will infer the target column, clean the dataset, evaluate multiple models, and produce a ranked
          report automatically.
        </p>
      </section>

      <form className="card upload-card" onSubmit={handleSubmit}>
        <label className="upload-zone">
          <input
            accept=".csv,.tsv,.txt"
            onChange={(event) => setFile(event.target.files?.[0] || null)}
            type="file"
          />
          <strong>{file ? file.name : "Choose or drop a CSV/TSV dataset"}</strong>
          <span>{file ? `${Math.round(file.size / 1024)} KB selected` : "Messy data is fine. The cleaner will handle it."}</span>
        </label>

        <div className="upload-meta">
          <div className="meta-card">
            <span>Agent provider</span>
            <strong>{settings.provider || "none"}</strong>
          </div>
          <div className="meta-card">
            <span>Primary model</span>
            <strong>{settings.model || "not set"}</strong>
          </div>
        </div>

        {error ? <p className="error-text">{error}</p> : null}
        <button className="button" disabled={submitting} type="submit">
          {submitting ? "Submitting..." : "Start autonomous run"}
        </button>
      </form>
    </div>
  );
}
