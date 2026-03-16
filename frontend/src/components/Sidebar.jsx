import { NavLink } from "react-router-dom";

const links = [
  { to: "/setup", label: "Setup Wizard" },
  { to: "/downloads", label: "Downloads" },
  { to: "/", label: "Dashboard" },
  { to: "/models", label: "Model Selector" },
  { to: "/upload", label: "Upload Dataset" }
];

export default function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <div className="brand-badge">ML</div>
        <div>
          <strong>Agent Console</strong>
          <p>Local, inspectable, and fast</p>
        </div>
      </div>
      <nav>
        {links.map((link) => (
          <NavLink
            key={link.to}
            className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}
            to={link.to}
          >
            {link.label}
          </NavLink>
        ))}
      </nav>
    </aside>
  );
}
