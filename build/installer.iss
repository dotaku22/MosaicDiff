; Per-user installer. No administrator prompt.
; Compile with: ISCC.exe build\installer.iss
; from the MosaicDiff folder, after PyInstaller has written dist\MosaicDiff.

#define AppName "MosaicDiff"
#define AppVersion "0.1.0"

[Setup]
AppId={{8F3A1C2E-6B47-4D19-9A55-7C2E0B4D91A6}
AppName={#AppName}
AppVersion={#AppVersion}
DefaultDirName={localappdata}\MosaicDiff
DefaultGroupName={#AppName}
PrivilegesRequired=lowest
OutputDir=..\dist
OutputBaseFilename=MosaicDiffSetup
Compression=lzma2
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern
UninstallDisplayIcon={app}\MosaicDiff.exe

[Files]
Source: "..\dist\MosaicDiff\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion
Source: "..\models\*"; DestDir: "{app}\models"; Flags: recursesubdirs ignoreversion
Source: "..\comfy_nodes\*"; DestDir: "{app}\comfy_nodes"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{autoprograms}\MosaicDiff"; Filename: "{app}\MosaicDiff.exe"
Name: "{autodesktop}\MosaicDiff"; Filename: "{app}\MosaicDiff.exe"

[Run]
Filename: "{app}\MosaicDiff.exe"; Description: "Open MosaicDiff"; Flags: postinstall nowait skipifsilent
