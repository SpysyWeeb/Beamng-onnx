-- Beamng-onnx in-game control panel.
--
-- Draws a small imgui window with ENGAGE / lane-change / turn / LONG /
-- CAL buttons and relays clicks as UDP datagrams to the Python control
-- panel (tools/control_panel.py) listening on 127.0.0.1:64257. The
-- Python side sends short status strings back, shown at the top.
--
-- The heavy lifting (camera -> onnx model -> controllers) stays in the
-- external process; this is UI only. Loaded by the control panel via
-- extensions.load('onnxPanel'), or manually from the console.

local M = {}

local im = ui_imgui
local socket = require('socket')

local PORT = 64257
local udp = nil
local statusText = 'waiting for control panel ...'
local engaged = false
local lastHello = 0

local function ensureSocket()
  if udp ~= nil then return end
  local ok, sock = pcall(socket.udp)
  if not ok or sock == nil then return end
  sock:settimeout(0)
  local okc = pcall(function() sock:setpeername('127.0.0.1', PORT) end)
  if not okc then return end
  udp = sock
end

local function send(cmd)
  ensureSocket()
  if udp then pcall(function() udp:send(cmd) end) end
end

local function onUpdate(dt)
  ensureSocket()
  if udp then
    -- announce ourselves so the panel learns our address
    lastHello = lastHello + (dt or 0)
    if lastHello > 2.0 then
      lastHello = 0
      send('hello')
    end
    while true do
      local data = udp:receive()
      if data == nil then break end
      statusText = data
      engaged = (string.sub(data, 1, 7) == 'ENGAGED')
    end
  end

  im.SetNextWindowSize(im.ImVec2(250, 240), im.Cond_FirstUseEver)
  im.Begin('ONNX Panel')
  im.TextUnformatted(statusText)
  im.Separator()
  local wide = im.ImVec2(226, 40)
  local half = im.ImVec2(110, 34)
  if engaged then
    if im.Button('DISENGAGE', wide) then send('engage') end
  else
    if im.Button('ENGAGE', wide) then send('engage') end
  end
  if im.Button('< LANE', half) then send('lane_l') end
  im.SameLine()
  if im.Button('LANE >', half) then send('lane_r') end
  if im.Button('< TURN', half) then send('turn_l') end
  im.SameLine()
  if im.Button('TURN >', half) then send('turn_r') end
  if im.Button('LONG', half) then send('long') end
  im.SameLine()
  if im.Button('CAL', half) then send('cal') end
  im.End()
end

local function onExtensionLoaded()
  ensureSocket()
  send('hello')
  log('I', 'onnxPanel', 'in-game control panel loaded (UDP ' .. PORT .. ')')
end

M.onUpdate = onUpdate
M.onExtensionLoaded = onExtensionLoaded

return M
