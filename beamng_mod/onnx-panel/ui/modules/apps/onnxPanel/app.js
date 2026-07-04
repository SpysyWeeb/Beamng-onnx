angular.module('beamng.apps')

.directive('onnxPanel', [function () {
  return {
    templateUrl: '/ui/modules/apps/onnxPanel/app.html',
    replace: true,
    restrict: 'EA',
    scope: true,
    link: function (scope, element, attrs) {
      scope.status = 'waiting for control panel ...';
      scope.engaged = false;
      scope.longMode = '?';

      scope.send = function (cmd) {
        bngApi.engineLua('if onnxPanel then onnxPanel.send("' + cmd + '") end');
      };

      // Hold-to-keep for desires: while a lane/turn button is held,
      // resend the command every 300 ms — the panel keeps the desire
      // (and blinker) alive ~0.8 s past the last repeat. A tap still
      // gets the normal one-shot pulse.
      var holdTimer = null;
      scope.desireDown = function (cmd) {
        scope.send(cmd);
        if (holdTimer) clearInterval(holdTimer);
        holdTimer = setInterval(function () { scope.send(cmd); }, 300);
      };
      scope.desireUp = function () {
        if (holdTimer) { clearInterval(holdTimer); holdTimer = null; }
      };

      // Gas/Brake nudge: while held, resend every 200 ms so the panel
      // keeps applying 30% to that pedal (its window expires ~0.35 s
      // after the last repeat). Release stops it.
      var pedalTimer = null;
      scope.pedalDown = function (cmd) {
        scope.send(cmd);
        if (pedalTimer) clearInterval(pedalTimer);
        pedalTimer = setInterval(function () { scope.send(cmd); }, 200);
      };
      scope.pedalUp = function () {
        if (pedalTimer) { clearInterval(pedalTimer); pedalTimer = null; }
      };
      scope.$on('$destroy', function () { scope.desireUp(); scope.pedalUp(); });

      // Status pushed by lua/ge/extensions/onnxPanel.lua via guihooks.
      scope.$on('OnnxPanelStatus', function (event, data) {
        scope.$evalAsync(function () {
          scope.status = data.text;
          scope.engaged = !!data.engaged;
          var m = /long (\w+)/.exec(data.text);
          scope.longMode = m ? m[1].toUpperCase() : '?';
        });
      });

      // The GE extension relays UDP to the external control panel —
      // make sure it's loaded even if this app is added before the
      // panel connects.
      bngApi.engineLua('extensions.load("onnxPanel")');
    }
  };
}]);
