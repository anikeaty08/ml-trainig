import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";

import ProgressBar from "../components/ProgressBar";

const API_URL =
  process.env.REACT_APP_API_URL ||
  (typeof window !== "undefined" ? window.location.origin : "http://127.0.0.1:38475");

export default function Processing() {
  const navigate = useNavigate();
  const { jobId } = useParams();
  const [status, setStatus] = useState({
    progress: 0,
    stage: "queued",
    message: "Waiting for worker"
  });

  useEffect(() => {
    const ws = new WebSocket(`${API_URL.replace("http", "ws")}/api/jobs/${jobId}/progress`);
    ws.onmessage = (event) => {
      const payload = JSON.parse(event.data);
      setStatus(payload);
      if (payload.stage === "completed") {
        window.setTimeout(() => navigate(`/results/${jobId}`), 700);
      }
    };
    return () => ws.close();
  }, [jobId, navigate]);

  return (
    <div className="stack">
      <section className="card hero-card">
        <div>
          <p className="eyebrow">Live execution</p>
          <h2>{status.stage}</h2>
          <p>{status.message}</p>
        </div>
        <Link className="button secondary" to="/">
          Back to dashboard
        </Link>
      </section>
      <section className="card">
        <ProgressBar label={status.stage} value={status.progress || 0} />
        {status.current_model ? (
          <div className="processing-details">
            <p>Current model: {status.current_model}</p>
            <p>
              Completed models: {status.completed_models} / {status.total_models}
            </p>
            {status.training_strategy ? <p>Strategy: {status.training_strategy}</p> : null}
            {status.architecture ? <p>Layers: {Array.isArray(status.architecture) ? status.architecture.join(" -> ") : status.architecture}</p> : null}
          </div>
        ) : null}
        {status.error_message ? <p className="error-text">{status.error_message}</p> : null}
      </section>
    </div>
  );
}
