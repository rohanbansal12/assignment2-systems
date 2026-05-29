# Basics
if command -v sudo >/dev/null; then SUDO=sudo; else SUDO=; fi
export DEBIAN_FRONTEND=noninteractive
$SUDO apt-get update
$SUDO apt-get install -y \
  curl \
  git \
  ca-certificates \
  build-essential \
  latexmk \
  texlive-latex-recommended \
  texlive-latex-extra \
  texlive-fonts-recommended \
  texlive-science

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Install uv globally for CS336 Python environment management.
curl -LsSf https://astral.sh/uv/install.sh -o /tmp/uv-install.sh
$SUDO env UV_INSTALL_DIR=/usr/local/bin sh /tmp/uv-install.sh
rm -f /tmp/uv-install.sh
hash -r

which uv
uv --version
which uvx
uvx --version

# Install and verify the CS336 assignment 2 environment.
cd "$SCRIPT_DIR"
uv sync
uv run python -c "import cs336_basics, cs336_systems; print('CS336 assignment2 uv environment ready')"

# Verify LaTeX tooling for writeup.pdf.
which latexmk
latexmk -version
which pdflatex
pdflatex --version | head -n 1

# Install nvm + Node LTS
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.4/install.sh | bash

export NVM_DIR="$HOME/.nvm"
. "$NVM_DIR/nvm.sh"

nvm install --lts
nvm use --lts
nvm alias default 'lts/*'

node -v
npm -v

# install Codex via npm
npm install -g @openai/codex@latest

# refresh shell command lookup
hash -r
