export default function ProgressBar({ value, label }) {
  return (
    <div className="progress-shell">
      <div className="progress-copy">
        <span>{label}</span>
        <strong>{value}%</strong>
      </div>
      <div className="progress-track">
        <div className="progress-fill" style={{ width: `${value}%` }} />
      </div>
    </div>
  );
}
