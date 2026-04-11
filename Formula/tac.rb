# Homebrew formula for tac - Task-Aware Codec
#
# This is a sketch formula. To publish to Homebrew:
# 1. Build the sdist/wheel and upload to PyPI
# 2. Update the url and sha256 below with the PyPI release URL
# 3. Submit to homebrew-core or host in a custom tap
#
# For a custom tap:
#   brew tap adpena/tap https://github.com/adpena/homebrew-tap
#   brew install adpena/tap/tac

class Tac < Formula
  include Language::Python::Virtualenv

  desc "Task-Aware Codec: Neural video compression for perception models"
  homepage "https://github.com/adpena/pact"
  url "https://files.pythonhosted.org/packages/source/t/tac/tac-1.0.0.tar.gz"
  sha256 "PLACEHOLDER_SHA256"
  license "MIT"

  depends_on "python@3.12"

  # PyTorch is too large for Homebrew bottles -- users install via pip.
  # This formula installs the CLI wrapper; torch must be in the venv.
  resource "pydantic" do
    url "https://files.pythonhosted.org/packages/source/p/pydantic/pydantic-2.11.1.tar.gz"
    sha256 "PLACEHOLDER_SHA256"
  end

  resource "numpy" do
    url "https://files.pythonhosted.org/packages/source/n/numpy/numpy-2.2.4.tar.gz"
    sha256 "PLACEHOLDER_SHA256"
  end

  def install
    virtualenv_install_with_resources
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/tac --version 2>&1", 0)
    system bin/"tac", "--help"
  end
end
