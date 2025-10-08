## ⚙️ Installation and Setup

This application requires **Python 3.8+** and uses **Poetry** for dependency management. Follow these steps to get the app running locally.

### 1\. Prerequisites

First, ensure you have **Python 3.8 or later** installed. You can check your version:

```bash
python --version
```

### 2\. Install Poetry

If you don't have Poetry, install it using the recommended method for your system by copying the following command and run it on your command line

**On macOS/Linux/WSL:**

```bash
curl -sSL https://install.python-poetry.org | python -
```

**On Windows (PowerShell):**

```bash
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
```

You may need to follow any on-screen prompts from the installer to add Poetry to your system's PATH.

-----

### 3\. Clone the Repository

Get a copy of the project files to your local machine:

```bash
git clone https://github.com/GentRoyal/datathon.git
cd datathon
```

-----

### 4\. Install Dependencies

Use Poetry to install all the necessary project dependencies:

```bash
poetry install
```

-----

### 5\. Run the Application

Start the web server using the `uvicorn` command, managed through Poetry:

```bash
poetry run uvicorn main:app --reload
```

You'll see output confirming the server is running:

```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

Access the application in your browser at: **[http://127.0.0.1:8000](http://127.0.0.1:8000)**



---

## ✨ Features

This application is currently under active development. Here are the main components and their current working status:

### ✅ Working Features

These features are fully implemented and available to use upon running the application:

* **Knowledge Coach**:
* **Lesson Copilot**: