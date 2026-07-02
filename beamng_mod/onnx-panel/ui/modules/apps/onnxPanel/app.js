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
