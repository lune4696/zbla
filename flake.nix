{
  description = "Zig development environment with Nix flakes";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/release-24.11";
    nixpkgs-unstable.url = "github:NixOS/nixpkgs/nixos-unstable";
    zig-overlay = {
      url = "github:mitchellh/zig-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, nixpkgs-unstable, zig-overlay }: 
    let
      systems = [
        "aarch64-darwin"
        "aarch64-linux"
        "x86_64-darwin"
        "x86_64-linux"
      ];
      forAllSystems = nixpkgs.lib.genAttrs systems;

    in {
     # for nix develop command
      devShells = forAllSystems (system: 
        let
          overlays = [zig-overlay.overlays.default];
          pkgs = import nixpkgs { inherit system overlays; };
          pkgs-unstable = import nixpkgs-unstable { inherit system overlays; };
          stable-pkgs = with pkgs; [
            # 基本ライブラリ
            stdenv
            pkgconf
            # 線形演算ライブラリ
            blas
            lapack
            # ユーティリティ
            cloc              # 行数カウンタ
            fzf               # fuzzy finder
            glxinfo           # GPU状態チェック
            helix             # editor
            valgrind          # メモリリークチェック
          ];
          unstable-pkgs = with pkgs-unstable; [
            # 言語別ライブラリ
            zig
            zls
          ];
        in {
          default = pkgs.mkShell {

            packages = stable-pkgs ++ unstable-pkgs;

            NIX_LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [ pkgs.pkgconf pkgs.raylib ];
            LD_LIBRARY_PATH = "$(nix-build '<nixpkgs>' -A wayland)/lib";

            # 環境変数設定 (オプション)
            ZIG_CACHE_DIR = "./.zig-cache"; # Zigのビルドキャッシュディレクトリ
            ZIG_GLOBAL_CACHE_DIR = "./.global_cache"; # グローバルキャッシュ

            shellHook = ''
              export PATH=${self}/bin:$PATH
              export PS1="\n⛄\[\033[1;32m\][\[\e]0;\u@\h: \w\a\]\u@\h:\w]\$ \[\033[0m\]"
              clear
              echo -e "\nWelcome to zig devShell!"
            '';
          };
        }
      );

    }; 
}

