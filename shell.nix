{pkgs ? (import <nixpkgs> {config.allowUnfree = true;})}: let
  mach-nix =
    import (builtins.fetchGit {
      url = "https://github.com/DavHau/mach-nix";
      rev = "8d903072c7b5426d90bc42a008242c76590af916";
    }) {
      python = "python310";
      pypiDataRev = "ba35683c35218acb5258b69a9916994979dc73a9";
      pypiDataSha256 = "sha256:019m7vnykryf0lkqdfd7sgchmfffxij1vw2gvi5l15an8z3rfi2p";
    };
  python = mach-nix.mkPython {
    requirements = builtins.readFile ./requirements.txt;
    _.nvidia-cufft-cu11.postInstall = "rm $out/lib/python*/site-packages/nvidia/__pycache__/__init__.cpython-310.pyc";
    _.nvidia-cuda-nvrtc-cu11.postInstall = "rm $out/lib/python*/site-packages/nvidia/__pycache__/__init__.cpython-310.pyc";
    _.nvidia-cudnn-cu11.postInstall = "rm $out/lib/python*/site-packages/nvidia/__pycache__/__init__.cpython-310.pyc";
  };
  tex = pkgs.texlive.combine { inherit (pkgs.texlive) scheme-full; };
  jupyenv = import (builtins.fetchGit {
    url = "https://github.com/tweag/jupyenv";
    rev = "3ad2c9512c9efd586cf63adde454e734a8ce049c";
  });
  inherit (jupyenv.lib.x86_64-linux) mkJupyterlabNew;
  jupyter = mkJupyterlabNew {
    kernel.python.minimal.enable = true;
  };
in
  pkgs.mkShell {
    buildInputs = with pkgs; [ tex python black jupyter ];
    shellHook = ''
      export CUDA_PATH=${pkgs.cudaPackages_11_7.cudatoolkit}
      export CUDA_HOME=${pkgs.cudaPackages_11_7.cudatoolkit}
      export LD_LIBRARY_PATH="${pkgs.cudaPackages_11_7.cudatoolkit}/lib"
      export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${pkgs.cudaPackages_11_7.cudnn}/lib"
    '';
  }
