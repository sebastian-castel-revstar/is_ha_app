
# Project Setup Instructions

This README outlines the necessary steps to set up the project environment. It includes instructions for installing essential tools and configuring pre-commit hooks as defined in the `.pre-commit-config.yaml` file for maintaining code quality and consistency.

## Prerequisites

Ensure Python is installed on your system, as the following steps require Python-based tools.

## Setup Steps

1. **Install CFN-Lint**:
   CFN-Lint is a tool to validate AWS CloudFormation YAML and JSON templates, helping to catch syntax errors, misconfigured attributes, and other common template issues.
   ```bash
   pip install cfn-lint
   ```

2. **Install Pre-Commit**:
   Pre-commit is a framework for managing multi-language pre-commit hooks. These hooks run checks before a commit is made, improving code quality.
   ```bash
   pip install pre-commit
   ```

3. **Clone the Repository**:
   Clone the project repository to your local machine. Replace `<repo url>` with the actual URL of the repository.
   ```bash
   git clone <repo url>
   ```

4. **Navigate to the Repository**:
   Change the directory to the cloned repository's root. Replace `<repo name>` with the name of the repository directory.
   ```bash
   cd <repo name>
   ```

5. **Install Pre-Commit Hooks**:
   Install the pre-commit hooks defined in the `.pre-commit-config.yaml`. These hooks ensure adherence to code standards.
   ```bash
   pre-commit install
   ```

## Pre-Commit Hooks Configuration

The `.pre-commit-config.yaml` includes several hooks for ensuring code quality:

- **Pre-commit-hooks**: A set of hooks by pre-commit to fix common issues like mixed line endings and trailing whitespaces.
- **Black**: A formatter for Python code, ensuring consistent coding style.
- **Script-must-have-extension**: Enforces that shell scripts have a `.sh` extension.
- **Ruff**: Python linter and formatter for maintaining code quality.
- **CFN-Lint**: Validates CloudFormation templates for syntax and best practices.

Each hook serves a specific purpose, contributing to the overall code quality and consistency of the project.

## Conclusion

By following these instructions, you will create a robust development environment for this project. The configured pre-commit hooks help maintain code standards, ensuring quality contributions.
