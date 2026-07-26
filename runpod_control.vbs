Option Explicit

Dim fso, shell, projectRoot, pythonw, command
Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")

projectRoot = fso.GetParentFolderName(WScript.ScriptFullName)
pythonw = fso.BuildPath(projectRoot, ".venv\Scripts\pythonw.exe")

If Not fso.FileExists(pythonw) Then
    MsgBox "AlphaSho virtual environment was not found:" & vbCrLf & pythonw, _
        vbCritical, "AlphaSho RunPod Control"
    WScript.Quit 1
End If

shell.CurrentDirectory = projectRoot
command = Chr(34) & pythonw & Chr(34) & " -m alphasho.runpod_control_app"
shell.Run command, 0, False
