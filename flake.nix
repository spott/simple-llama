{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-22.11-darwin";
  };

  outputs = { self, nixpkgs }: 
  let
    system = "aarch64-darwin";
    overlays = [
        (self: super: {
          python = super.python310;
        })
      ];
    pkgs = import nixpkgs { inherit system overlays; };
  in
  {
    devShells.${system}.default = pkgs.mkShell {
        buildInputs = with pkgs; [ python ] ++ 
          (with pkgs.python310Packages; [ pipenv ]);
      };
  };
}
