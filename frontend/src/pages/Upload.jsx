import { useState } from "react";
import { useNavigate } from "react-router-dom";

import { api } from "../services/api";

export default function Upload({ settings }) {
  const navigate = useNavigate();
  const [file, setFile] = useState(null);
  const [datasetUrl, setDatasetUrl] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");

  async function handleFileSubmit(event) {
    event.preventDefault();
    if (!file) {
      setError("Choose a dataset file first.");
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

  async function handleUrlSubmit(event) {
    event.preventDefault();
    if (!datasetUrl.trim()) {
      setError("Paste a dataset URL first.");
      return;
    }

    setSubmitting(true);
    setError("");
    try {
      const payload = await api.submitJobFromUrl({ url: datasetUrl.trim(), settings });
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
          The pipeline will infer the target column, decide whether the dataset is tabular, text-heavy, or time-series,
          then clean it, evaluate multiple models, and produce a ranked report automatically.
        </p>
        <div className="setup-steps">
          <strong>Accepted dataset sources</strong>
          <p>Local files: `.csv`, `.tsv`, `.txt`, and `.zip` archives.</p>
          <p>Direct URLs: paste a raw download URL.</p>
          <p>Kaggle: paste a dataset URL, a competition URL, a bare ref like `owner/dataset`, or a `kagglehub.dataset_download(...)` snippet.</p>
        </div>
      </section>

      <form className="card upload-card" onSubmit={handleFileSubmit}>
        <label className="upload-zone">
          <input
            accept=".csv,.tsv,.txt,.zip"
            onChange={(event) => setFile(event.target.files?.[0] || null)}
            type="file"
          />
          <strong>{file ? file.name : "Choose or drop a CSV/TSV dataset"}</strong>
          <span>
            {file
              ? `${Math.round(file.size / 1024)} KB selected`
              : "CSV/TSV/TXT for tabular, text, and time-series data, or ZIP archives for image/audio datasets."}
          </span>
        </label>

        <div className="upload-meta">
          <div className="meta-card">
            <span>Agent provider</span>
            <strong>{settings.provider || "none"}</strong>
          </div>
          <div className="meta-card">
            <span>Primary model ref</span>
            <strong>{settings.primary_model_ref || "not set"}</strong>
          </div>
        </div>

        {error ? <p className="error-text">{error}</p> : null}
        <button className="button" disabled={submitting} type="submit">
          {submitting ? "Submitting..." : "Start autonomous run"}
        </button>
      </form>

      <form className="card upload-card" onSubmit={handleUrlSubmit}>
        <p className="eyebrow">Dataset URL</p>
        <h3>Pull from URL, Kaggle, or KaggleHub ref</h3>
        <p>
          Paste a direct dataset URL, a Kaggle dataset/competition link, a bare Kaggle ref like
          `rhythmghai/300k-student-performance-prediction-dataset`, or the full `kagglehub.dataset_download(...)`
          snippet. The app will download it locally first, then run the same autonomous pipeline.
        </p>
        <label>
          Dataset URL
          <input
            onChange={(event) => setDatasetUrl(event.target.value)}
            placeholder='https://..., https://www.kaggle.com/datasets/..., owner/dataset, or kagglehub.dataset_download("owner/dataset")'
            type="text"
            value={datasetUrl}
          />
        </label>
        <button className="button secondary" disabled={submitting} type="submit">
          {submitting ? "Fetching..." : "Fetch and train"}
        </button>
      </form>
    </div>
  );
}
