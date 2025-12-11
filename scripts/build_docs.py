#!/usr/bin/env python3
"""
Build and optionally serve the documentation.
"""

import os
import subprocess
import sys
import webbrowser
from pathlib import Path
import argparse


def build_docs(clean=False, builder="html"):
    """Build the documentation using Sphinx."""
    
    project_root = Path(__file__).parent.parent
    docs_dir = project_root / "docs"
    build_dir = docs_dir / "build"
    
    os.chdir(docs_dir)
    
    # Clean build directory if requested
    if clean and build_dir.exists():
        print("Cleaning build directory...")
        subprocess.run(["rm", "-rf", str(build_dir)], check=True)
    
    # Run sphinx-build
    print(f"Building documentation ({builder})...")
    cmd = [
        "sphinx-build",
        "-b", builder,
        "-W",  # Turn warnings into errors
        "--keep-going",  # Continue on errors
        "source",
        f"build/{builder}"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Documentation built successfully!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error building docs: {e}")
        if e.stderr:
            print(e.stderr)
        if e.stdout:
            print(e.stdout)
        return False


def serve_docs(port=8000):
    """Serve the documentation locally."""
    
    project_root = Path(__file__).parent.parent
    html_dir = project_root / "docs" / "build" / "html"
    
    if not html_dir.exists():
        print("Documentation not built yet. Run with --build first.")
        sys.exit(1)
    
    print(f"Serving documentation at http://localhost:{port}")
    print("   Press Ctrl+C to stop")
    
    # Open browser
    webbrowser.open(f"http://localhost:{port}")
    
    # Start server
    os.chdir(html_dir)
    subprocess.run(["python3", "-m", "http.server", str(port)])


def main():
    parser = argparse.ArgumentParser(
        description="Build and serve SheetWise documentation"
    )
    parser.add_argument(
        "--build", "-b",
        action="store_true",
        help="Build the documentation"
    )
    parser.add_argument(
        "--clean", "-c",
        action="store_true",
        help="Clean build directory before building"
    )
    parser.add_argument(
        "--serve", "-s",
        action="store_true",
        help="Serve the documentation locally"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8000,
        help="Port for local server (default: 8000)"
    )
    parser.add_argument(
        "--builder",
        choices=["html", "latex", "linkcheck"],
        default="html",
        help="Sphinx builder to use (default: html)"
    )
    
    args = parser.parse_args()
    
    # If no arguments, build and serve
    if not (args.build or args.serve):
        args.build = True
        args.serve = True
    
    if args.build:
        success = build_docs(clean=args.clean, builder=args.builder)
        if not success and args.serve:
            print("Cannot serve documentation due to build errors")
            sys.exit(1)
    
    if args.serve:
        serve_docs(port=args.port)


if __name__ == "__main__":
    main()
