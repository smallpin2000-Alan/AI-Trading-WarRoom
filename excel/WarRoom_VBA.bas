Attribute VB_Name = "WarRoom_VBA"
' =========================================================================
' AI 智慧看盤與自動交易戰情室 — Excel VBA 巨集橋接模組
' 支援：
' 1. 單次手動刷新 (RefreshWarRoom_Once)
' 2. 定時自動刷新 (StartAutoRefresh / StopAutoRefresh)
' 3. 緊急市價平倉 (FlattenNow)
' 4. 開啟/關閉自動操盤 (ToggleAutoPilot)
' =========================================================================

Option Explicit

Private NextRefreshTime As Double
Private IsAutoRunning As Boolean

' 【手動單次刷新】
Sub RefreshWarRoom_Once()
    On Error GoTo ErrorHandler
    Dim wsh As Object
    Dim cmd As String
    Dim projectPath As String
    
    ' 自動取得當前活頁簿所在專案資料夾路徑
    projectPath = ThisWorkbook.Path
    If projectPath = "" Then
        projectPath = "G:\我的雲端硬碟\工具\自動測試"
    End If
    
    cmd = "cmd.exe /c cd /d """ & projectPath & """ && python -m src.ui.excel_warroom --once"
    
    Set wsh = CreateObject("WScript.Shell")
    ' 0 代表完全靜默於背景執行視窗，True 代表等待執行完畢
    wsh.Run cmd, 0, True
    
    Application.StatusBar = "【戰情室】已於 " & Format(Now, "hh:nn:ss") & " 完成最新即時數據同步"
    Exit Sub

ErrorHandler:
    MsgBox "背景刷新失敗: " & Err.Description, vbCritical, "Excel 戰情室錯誤"
End Sub

' 【啟動定時自動刷新】
Sub StartAutoRefresh()
    Dim intervalSec As Integer
    Dim ws As Worksheet
    
    Set ws = ThisWorkbook.Worksheets("📊 戰情室即時看盤")
    intervalSec = CInt(ws.Range("F5").Value)
    If intervalSec <= 0 Then intervalSec = 10
    
    IsAutoRunning = True
    Call RefreshWarRoom_Once
    
    NextRefreshTime = Now + TimeSerial(0, 0, intervalSec)
    Application.OnTime NextRefreshTime, "AutoRefreshLoop"
    Application.StatusBar = "【戰情室】定時自動監控已啟動 (每 " & intervalSec & " 秒刷新一次)"
End Sub

' 【自動刷新迴圈回呼函式】
Sub AutoRefreshLoop()
    If Not IsAutoRunning Then Exit Sub
    
    Dim intervalSec As Integer
    Dim ws As Worksheet
    
    Set ws = ThisWorkbook.Worksheets("📊 戰情室即時看盤")
    intervalSec = CInt(ws.Range("F5").Value)
    If intervalSec <= 0 Then intervalSec = 10
    
    Call RefreshWarRoom_Once
    
    NextRefreshTime = Now + TimeSerial(0, 0, intervalSec)
    Application.OnTime NextRefreshTime, "AutoRefreshLoop"
End Sub

' 【停止定時自動刷新】
Sub StopAutoRefresh()
    IsAutoRunning = False
    On Error Resume Next
    If NextRefreshTime > 0 Then
        Application.OnTime NextRefreshTime, "AutoRefreshLoop", , False
    End If
    On Error GoTo 0
    Application.StatusBar = "【戰情室】定時自動監控已停止"
    MsgBox "已停止自動更新定時器。", vbInformation, "戰情室狀態"
End Sub

' 【緊急手動市價平倉】
Sub FlattenNow()
    Dim ans As VbMsgBoxResult
    ans = MsgBox("確定要對模擬倉進行【手動緊急市價平倉】嗎？", vbQuestion + vbYesNo + vbDefaultButton2, "確認平倉")
    If ans = vbYes Then
        Dim ws As Worksheet
        Set ws = ThisWorkbook.Worksheets("📊 戰情室即時看盤")
        ws.Range("D5").Value = "平倉"
        Call RefreshWarRoom_Once
        MsgBox "平倉指令已送出並由系統執行平倉完畢！", vbInformation, "平倉成功"
    End If
End Sub

' 【切換自動操盤 ON / OFF】
Sub ToggleAutoPilot()
    Dim ws As Worksheet
    Set ws = ThisWorkbook.Worksheets("📊 戰情室即時看盤")
    Dim currState As String
    currState = UCase(Trim(ws.Range("B5").Value))
    
    If currState = "ON" Then
        ws.Range("B5").Value = "OFF"
        MsgBox "自動操盤功能已關閉 (OFF)。", vbExclamation, "操盤狀態"
    Else
        ws.Range("B5").Value = "ON"
        MsgBox "自動操盤功能已開啟 (ON)。", vbInformation, "操盤狀態"
    End If
    Call RefreshWarRoom_Once
End Sub
