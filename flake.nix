{
  description = "Custom clang";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = { self, nixpkgs, flake-utils, ...}:
  flake-utils.lib.eachDefaultSystem (system:
    let
      pkgs = import nixpkgs { inherit system; };
    in
    {

      defaultPackage = pkgs.llvmPackages.stdenv.mkDerivation {
        name = "custom-clang";
        src = self;
        nativeBuildInputs = pkgs.clang.nativeBuildInputs ++ [
          pkgs.cmake
          pkgs.ninja
          pkgs.gdb
        ];
        hardeningDisable = [ "all" ];
      };
    }
  );
}
