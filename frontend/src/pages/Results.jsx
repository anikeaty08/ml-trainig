import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";

import ResultsDisplay from "../components/ResultsDisplay";
import { api } from "../services/api";

export default function Results() {
  const { jobId } = useParams();
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const payload = await api.getJobResults(jobId);
        if (!cancelled) {
          setResult(payload);
        }
      } catch (loadError) {
        if (!cancelled) {
          setError(loadError.message);
        }
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, [jobId]);

  if (error) {
    return (
      <section className="card">
        <h2>Could not load results</h2>
        <p className="error-text">{error}</p>
      </section>
    );
  }

  if (!result) {
    return (
      <section className="card">
        <h2>Loading results...</h2>
      </section>
    );
  }

  return <ResultsDisplay jobId={jobId} result={result} />;
}
