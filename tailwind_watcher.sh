#!/bin/bash

# Script to run Tailwind CSS in watch mode

echo "Starting Tailwind CSS watcher..."

./tailwindcss -i ./static/css/input.css -o ./static/css/output.css --watch

echo "TailwindCSS watcher started."