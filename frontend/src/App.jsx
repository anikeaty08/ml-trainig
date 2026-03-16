import { startTransition, useEffect, useMemo, useState } from "react";
import { Navigate, Route, Routes, useLocation } from "react-router-dom";

import Header from "./components/Header";
import Sidebar from "./components/Sidebar";
import Downloads from "./pages/Downloads";
import Dashboard from "./pages/Dashboard";
import ModelSelection from "./pages/ModelSelection";
import Processing from "./pages/Processing";
import Results from "./pages/Results";
import Setup from "./pages/Setup";
import Terminal from "./pages/Terminal";
import Upload from "./pages/Upload";
import { usePolling } from "./hooks/usePolling";
import { api } from "./services/api";

const defaultSettings = {
  provider: "",
  auth_mode: "local",
  base_url: "",
  primary_model_ref: "",
  image_model_ref: "",
  model: "",
  fallback_models: "",
  fallback_model_refs: "",
  model_allowlist: "",
  model_catalog: [
    {
      alias: "local-fast",
      ref: "",
      capability: "chat",
      description: "Set this after choosing a provider"
    },
    {
      alias: "local-vision",
      ref: "",
      capability: "image",
      description: "Optional image-capable route"
    }
  ],
  auth_profiles: [],
  auth_order: {},
  browser_session_hint: "",
  api_key: ""
};

export default function App() {
  const location = useLocation();
  const [jobs, setJobs] = useState([]);
  const [availableModels, setAvailableModels] = useState({ builtin_models: [], candidate_families: [] });
  const [providerCatalog, setProviderCatalog] = useState([]);
  const [settings, setSettings] = useState(defaultSettings);
  const [configLoaded, setConfigLoaded] = useState(false);
  const [setupStatus, setSetupStatus] = useState(null);

  useEffect(() => {
    let cancelled = false;

    async function loadMeta() {
      const [models, providers, storedConfig, nextSetupStatus] = await Promise.all([
        api.getAvailableModels(),
        api.getProviders(),
        api.getStoredConfig(),
        api.getSetupStatus()
      ]);
      if (!cancelled) {
        setAvailableModels(models);
        setProviderCatalog(providers.providers || []);
        setSettings({ ...defaultSettings, ...storedConfig });
        setSetupStatus(nextSetupStatus);
        setConfigLoaded(true);
      }
    }

    loadMeta();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!configLoaded) {
      return;
    }
    const timer = window.setTimeout(() => {
      api.saveStoredConfig(settings).catch(() => {});
    }, 250);
    return () => window.clearTimeout(timer);
  }, [configLoaded, settings]);

  async function refreshJobs() {
    const data = await api.listJobs();
    startTransition(() => setJobs(data));
  }

  usePolling(refreshJobs, 3000);

  const activeModelRef = useMemo(() => {
    if (settings.primary_model_ref) {
      return settings.primary_model_ref;
    }
    if (!settings.model) {
      return settings.provider || "unset";
    }
    return settings.model.includes("/") ? settings.model : `${settings.provider}/${settings.model}`;
  }, [settings]);

  function handleSetupUpdate(nextStatus) {
    setSetupStatus(nextStatus);
  }

  const needsSetup = Boolean(setupStatus?.needs_setup);
  const shouldRedirectToSetup = needsSetup && location.pathname !== "/setup";

  return (
    <div className="app-shell">
      <Sidebar />
      <main className="main-panel">
        <Header activeModelRef={activeModelRef} jobCount={jobs.length} setupStatus={setupStatus} />
        <Routes>
          {shouldRedirectToSetup ? <Route element={<Navigate replace to="/setup" />} path="*" /> : null}
          <Route element={<Setup onSetupUpdate={handleSetupUpdate} providerCatalog={providerCatalog} setupStatus={setupStatus} />} path="/setup" />
          <Route element={<Downloads onSetupUpdate={handleSetupUpdate} setupStatus={setupStatus} />} path="/downloads" />
          <Route
            element={
              <Dashboard
                availableModels={availableModels}
                jobs={jobs}
                onJobsRefresh={refreshJobs}
                onSettingsChange={setSettings}
                providerCatalog={providerCatalog}
                settings={settings}
                setupStatus={setupStatus}
              />
            }
            path="/"
          />
          <Route
            element={
              <ModelSelection
                onSettingsChange={setSettings}
                providerCatalog={providerCatalog}
                settings={settings}
              />
            }
            path="/models"
          />
          <Route
            element={
              <Terminal
                currentJobId={jobs[0]?.id}
                onSettingsChange={setSettings}
                providerCatalog={providerCatalog}
                settings={settings}
              />
            }
            path="/terminal"
          />
          <Route element={<Upload settings={settings} />} path="/upload" />
          <Route element={<Processing />} path="/processing/:jobId" />
          <Route element={<Results />} path="/results/:jobId" />
        </Routes>
      </main>
    </div>
  );
}
