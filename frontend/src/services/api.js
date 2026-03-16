const API_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

async function request(path, options = {}) {
  const response = await fetch(`${API_URL}${path}`, {
    headers: {
      ...(options.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
      ...(options.headers || {})
    },
    ...options
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Request failed: ${response.status}`);
  }

  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    return response.json();
  }
  return response.text();
}

export const api = {
  getAvailableModels: () => request("/api/models/available"),
  listJobs: () => request("/api/jobs"),
  getJob: (jobId) => request(`/api/jobs/${jobId}`),
  getJobResults: (jobId) => request(`/api/jobs/${jobId}/results`),
  getProviders: () => request("/api/agent/providers"),
  getRemoteModels: (settings) =>
    request("/api/agent/models", {
      method: "POST",
      body: JSON.stringify(settings)
    }),
  runConsoleCommand: ({ command, settings, jobId }) =>
    request("/api/agent/command", {
      method: "POST",
      body: JSON.stringify({ command, settings, job_id: jobId })
    }),
  submitJob: ({ file, settings }) => {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("provider", settings.provider || "");
    formData.append("model", settings.primary_model_ref || settings.model || "");
    formData.append("settings_json", JSON.stringify(settings || {}));
    return request("/api/jobs/submit", {
      method: "POST",
      body: formData
    });
  },
  getDownloadUrl: (jobId, kind) => `${API_URL}/api/downloads/${jobId}/${kind}`
};
