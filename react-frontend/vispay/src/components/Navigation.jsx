import { NavLink } from "react-router-dom";
import { Shield, Search } from "lucide-react";

export default function Navigation() {
  return (
    <nav className="sticky top-0 z-50 bg-background border-b border-border">
      <div className="px-8 py-5 flex items-center justify-between">
        {/* Logo and Branding */}
        <div className="flex items-center gap-3">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <img
                src="/icons/main.png"
                alt="VisPay Logo"
                className="h-10 w-auto object-contain"
              />
            </div>
          </div>
          <div className="flex flex-col gap-0.5">
            <h1 className="text-lg font-bold text-foreground tracking-tight">
              VisPay Vision
            </h1>
            <p className="text-xs text-muted-foreground font-medium">
              Live Commerce Intelligence
            </p>
          </div>
        </div>

        {/* Navigation Links */}
        <div className="flex items-center gap-2">
          <NavLink
            to="/payment"
            className={({ isActive }) =>
              `flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
                isActive
                  ? "bg-primary text-primary-foreground"
                  : "text-foreground hover:bg-primary/10"
              }`
            }
          >
            <Shield className="w-4 h-4" />
            <span className="font-medium">Payment Verification</span>
          </NavLink>
          <NavLink
            to="/search"
            className={({ isActive }) =>
              `flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
                isActive
                  ? "bg-primary text-primary-foreground"
                  : "text-foreground hover:bg-primary/10"
              }`
            }
          >
            <Search className="w-4 h-4" />
            <span className="font-medium">Product Search</span>
          </NavLink>
        </div>
      </div>
    </nav>
  );
}
