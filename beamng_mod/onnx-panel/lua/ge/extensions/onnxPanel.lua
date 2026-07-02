-- Beamng-onnx in-game control panel: GE-side relay.
--
-- The visible panel is a proper UI app (ui/modules/apps/onnxPanel/ —
-- place it via the in-game UI Apps editor like a speedometer). This
-- extension is the plumbing between that app and the external Python
-- control panel (tools/control_panel.py):
--
--   UI app button -> bngApi.engineLua -> M.send() -> UDP 127.0.0.1:64257
--   Python status -> UDP -> onUpdate() -> guihooks 'OnnxPanelStatus' -> app
--
-- The heavy lifting (camera -> onnx model -> controllers) stays in the
-- external process; nothing here touches the vehicle.

local M = {}

local socket = require('socket')

local PORT = 64257
local udp = nil
local status = { text = 'waiting for control panel ...', engaged = false }
local helloTimer = 0

local function ensureSocket()
  if udp ~= nil then return end
  local ok, sock = pcall(socket.udp)
  if not ok or sock == nil then return end
  sock:settimeout(0)
  local okc = pcall(function() sock:setpeername('127.0.0.1', PORT) end)
  if not okc then return end
  udp = sock
end

function M.send(cmd)
  ensureSocket()
  if udp then pcall(function() udp:send(cmd) end) end
end

local function push()
  guihooks.trigger('OnnxPanelStatus', status)
end

local function onUpdate(dt)
  ensureSocket()
  if udp == nil then return end
  -- periodic hello so the panel learns/refreshes our address
  helloTimer = helloTimer + (dt or 0)
  if helloTimer > 2.0 then
    helloTimer = 0
    M.send('hello')
  end
  local changed = false
  while true do
    local data = udp:receive()
    if data == nil then break end
    status.text = data
    status.engaged = (string.sub(data, 1, 7) == 'ENGAGED')
    changed = true
  end
  if changed then push() end
end

local function onExtensionLoaded()
  ensureSocket()
  M.send('hello')
  push()
  log('I', 'onnxPanel', 'relay loaded (UDP ' .. PORT .. ')')
end

M.onUpdate = onUpdate
M.onExtensionLoaded = onExtensionLoaded

return M
