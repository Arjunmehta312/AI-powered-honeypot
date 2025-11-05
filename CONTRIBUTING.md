# Contributing to AI-Powered Honeypot Intelligence System

Thank you for your interest in contributing to this project!

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone <your-fork-url>`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Test your changes thoroughly
6. Commit your changes: `git commit -m "Description of changes"`
7. Push to your fork: `git push origin feature/your-feature-name`
8. Create a Pull Request

## Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install pytest black flake8 mypy
```

## Code Style

- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and concise
- Add comments for complex logic

## Testing

Before submitting a pull request:

1. Run the complete pipeline to ensure no errors
2. Test the dashboard functionality
3. Verify model training completes successfully
4. Check that documentation is updated

## Areas for Contribution

- **New Features**: Additional ML models, new visualizations, enhanced dashboard
- **Bug Fixes**: Report and fix bugs
- **Documentation**: Improve guides, add examples, fix typos
- **Performance**: Optimize processing speed, reduce memory usage
- **Testing**: Add unit tests, integration tests

## Reporting Issues

When reporting issues, please include:

- Python version
- Operating system
- Error messages (full traceback)
- Steps to reproduce
- Expected vs actual behavior

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Help others learn and grow
- Focus on what is best for the project

## Questions?

Open an issue with the "question" label or start a discussion.

Thank you for contributing!
