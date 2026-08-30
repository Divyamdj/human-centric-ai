# HCAI Project

This document provides instructions on how to set up and run this Django project.

## Prerequisites
- Python 3.10 to 3.12 (for manual setup)
- Docker & Docker Compose (for Docker setup)
- pip (Python package installer, for manual setup)

## Getting Started

You can set up and run the project in **two ways**:

---

### Option 1: Using Docker (Recommended)

1. **Build and Start the Containers**

   ```bash
   docker-compose up --build
   ```

   This will:
   - Build the Docker image (installing all dependencies)
   - Run database migrations
   - Start the Django development server at [http://localhost:8000/](http://localhost:8000/)

2. **Environment Variables**

   If you see a warning about a missing `.env` file, create one in your project root. Example:

   ```
   DJANGO_SECRET_KEY=your-secret-key
   DJANGO_DEBUG=True
   ```

   Add any other environment variables your project needs.

---

### Option 2: Manual Local Setup

1. **Create and Activate a Virtual Environment**

   **On macOS/Linux:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

   **On Windows:**
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   If you encounter issues installing the **gosdt** library, please refer to the special instructions in the section below.

3. **Run Database Migrations**

   ```bash
   python manage.py migrate
   ```

4. **Start the Development Server**

   ```bash
   python manage.py runserver
   ```

   The project will be running at [http://127.0.0.1:8000/](http://127.0.0.1:8000/).

---

> **Note**: The "Creating global recommender instance..." message may take time because the `MovieRecommender` class performs the following on initialization: loads all ratings and movies from CSV files (potentially thousands or millions of rows), creates a large user-movie matrix in memory, initializes random latent factor matrices for all users and movies, and runs computationally expensive matrix factorization (gradient descent) for many iterations over all observed ratings. This causes the Django server to load all data and retrain the recommender from scratch each time it starts, which can be slow for non-trivial datasets.

---

## Troubleshooting gosdt Installation

If `pip install gosdt` from the requirements.txt file fails, you may need to install it manually.

### For Windows Users (Simple Method)
1. Download **pkg-config-lite** from [https://sourceforge.net/projects/pkgconfiglite/](https://sourceforge.net/projects/pkgconfiglite/).
2. Extract the zip file.
3. Add the location of the `bin` folder to your system's environment variables.
4. Try installing the library again:
   ```bash
   pip install gosdt
   ```

---

## How to Build the Project from Source
If the above method does not work, you will need to build the library from source. Further information about the gosdt library and downloadable distributions can be found on its [PyPI page](https://pypi.org/project/gosdt/).

### Step 1: Install Required Development Tools

**macOS:**
```bash
brew install cmake ninja pkg-config
pip3 install --upgrade scikit-build-core pybind11 delocate
```

**Ubuntu:**
```bash
sudo apt install -y cmake ninja-build pkg-config patchelf
pip3 install --upgrade scikit-build-core pybind11 auditwheel
```

**Windows (Run Powershell as Admin):**

#### Step 1.1: Install Chocolatey
```powershell
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
```

#### Step 1.2: Install vcpkg
```powershell
cd C:\
git clone https://github.com/Microsoft/vcpkg.git
.\vcpkg\bootstrap-vcpkg.bat
```

Update PATH and set `VCPKG_INSTALLATION_ROOT`:
```powershell
$vcpkg = "C:\vcpkg"
$old = (Get-ItemProperty -Path 'Registry::HKEY_LOCAL_MACHINE\System\CurrentControlSet\Control\Session Manager\Environment' -Name PATH).path
$new = "$old;$vcpkg"
Set-ItemProperty -Path 'Registry::HKEY_LOCAL_MACHINE\System\CurrentControlSet\Control\Session Manager\Environment' -Name PATH -Value $new
Set-ItemProperty -Path 'Registry::HKEY_LOCAL_MACHINE\System\CurrentControlSet\Control\Session Manager\Environment' -Name VCPKG_INSTALLATION_ROOT -Value $vcpkg
```

#### Step 1.3: Install required development tools
```powershell
winget install Kitware.CMake
choco install -y ninja
choco install -y pkgconfiglite
pip3 install --upgrade scikit-build
pip3 install --upgrade delvewheel
```

### Step 2: Install Required 3rd-Party Libraries

**macOS:**
```bash
brew install tbb gmp
```

**Ubuntu:**
```bash
sudo apt install -y libtbb-dev libgmp-dev
```

**Windows:**
```powershell
vcpkg install tbb:x64-windows
vcpkg install gmp:x64-windows
```

### Step 3: Build the Project

**Method 1: Local use/development and debugging**
```bash
pip3 install .
```

**Method 2: Wheel generation**

**macOS:**
```bash
pip3 wheel --no-deps . -w dist/
delocate-wheel -w dist -v dist/gosdt-*.whl
```

**Windows:**
```powershell
pip3 wheel --no-deps . -w dist/
python3 -m delvewheel repair --no-mangle-all --add-path "$ENV:VCPKG_INSTALLATION_ROOT\installed\x64-windows\bin" dist/gosdt-*.whl -w dist
```

**Ubuntu (manylinux, using Docker + cibuildwheel):**
```bash
pipx run cibuildwheel
```

---

## Frontend Stack

This project uses a modern, lightweight frontend stack that enhances user interactivity without the complexity of a large JavaScript framework.

### Tailwind CSS
Tailwind CSS is a utility-first CSS framework used for styling the application. Instead of writing custom CSS, you build designs directly in your HTML by composing utility classes. This makes styling rapid, consistent, and maintainable.

### HTMX
HTMX allows you to access modern browser features like AJAX and CSS Transitions directly from HTML, without writing JavaScript. It is used to create dynamic and interactive user experiences, such as submitting forms without a full page reload or updating parts of a page based on user actions.

### Alpine.js
Alpine.js provides the reactive and declarative nature of larger frameworks like Vue or React at a much lower cost. It's used for managing small pieces of client-side interactivity and state, such as toggling dropdowns, managing tabs, or handling simple UI element states directly in your markup.