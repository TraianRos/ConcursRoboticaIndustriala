#!/usr/bin/env python3
"""Incremental DOFBOT gamepad control, based on the arm_handle notebook.

Uses Linux /dev/input/js*; connect the gamepad to the Pi. Each input jogs one servo
independently. No automatic homing or torque changes.
"""
import argparse
import fcntl
import json
from math import isfinite
from numbers import Real
import os
from pathlib import Path
import struct
import sys
import time

_EVENT = struct.Struct('=IhBB')
_HARD_MAX = (180, 180, 180, 180, 270, 180)


def number(value):
    return isinstance(value, Real) and not isinstance(value, bool) and isfinite(value)


class JogPlanner:
    """Generate bounded joint targets. No hardware access in this class."""
    def __init__(self, pose, limits, speed=12, deadzone=0.15):
        if not number(speed) or not 0 < speed <= 30:
            raise ValueError('speed_deg_s must be between 0 (exclusive) and 30')
        if not number(deadzone) or not 0 <= deadzone < 0.8:
            raise ValueError('deadzone must be between 0 and 0.8 (exclusive)')
        if len(limits) != 6:
            raise ValueError('limits must contain six [min, max] pairs')
        for joint, (pair, maximum) in enumerate(zip(limits, _HARD_MAX), 1):
            if (len(pair) != 2 or not all(number(v) and float(v).is_integer() for v in pair)
                    or not 0 <= pair[0] < pair[1] <= maximum):
                raise ValueError('invalid limits for joint {}'.format(joint))
        self.limits = tuple(tuple(pair) for pair in limits)
        self.speed, self.deadzone = speed, deadzone
        self.reset(pose)

    def check_pose(self, pose):
        if len(pose) != 6:
            raise ValueError('expected six servo readings')
        for joint, (angle, maximum) in enumerate(zip(pose, _HARD_MAX), 1):
            if not number(angle):
                raise ValueError('joint {} reading {!r} is unavailable or invalid'.format(joint,angle))
            if not 0 <= angle <= maximum:
                raise ValueError('joint {} reading {} is outside hardware range [0, {}]'
                                 .format(joint,angle,maximum))

    def limit_warnings(self, pose):
        self.check_pose(pose)
        return ['Servo {} is at {} degrees, outside [{}, {}]; only inward jogging is allowed.'
                .format(joint,angle,lower,upper)
                for joint,(angle,(lower,upper)) in enumerate(zip(pose,self.limits),1)
                if not lower <= angle <= upper]

    def reset(self, pose):
        self.check_pose(pose)
        self.target = list(map(float, pose))

    def _axis(self, value):
        if not number(value) or not -1 <= value <= 1:
            raise ValueError('invalid gamepad axis value')
        if abs(value) <= self.deadzone:
            return 0.0
        return (1 if value > 0 else -1) * (abs(value)-self.deadzone)/(1-self.deadzone)

    def step(self, inputs, dt):
        """Six independent signed inputs, one per servo; no compensation."""
        if len(inputs) != 6:
            raise ValueError('expected six servo inputs')
        if not number(dt) or dt < 0:
            raise ValueError('invalid elapsed time')
        amount = self.speed * min(dt, 0.15)
        delta = [self._axis(value)*amount for value in inputs]
        # Saturate each servo separately. One joint at its limit cannot freeze
        # another servo, and returning the input reverses away from the limit.
        targets = []
        for angle,change,(lower,upper) in zip(self.target,delta,self.limits):
            if angle < lower:
                # An existing out-of-margin position may return gradually;
                # neutral/outward input never snaps it to the minimum.
                target = min(lower,angle+change) if change > 0 else angle
            elif angle > upper:
                target = max(upper,angle+change) if change < 0 else angle
            else:
                target = max(lower,min(upper,angle+change))
            targets.append(target)
        self.target = targets
        return tuple(int(round(angle)) for angle in self.target)


def _index(value):
    if type(value) is not int or value < 0:
        raise ValueError('binding indices must be nonnegative integers')
    return value


def binding_resources(spec, signed):
    """Validate a binding and list the raw inputs it uses."""
    kind = spec['type']
    if kind not in (('axis', 'buttons') if signed else ('trigger', 'button')):
        raise ValueError('invalid input type: {}'.format(kind))
    if signed:
        if type(spec.get('sign', 1)) is not int or spec.get('sign', 1) not in (-1, 1):
            raise ValueError('servo binding sign must be -1 or 1')
    if kind == 'buttons':
        negative, positive = _index(spec['negative']), _index(spec['positive'])
        if negative == positive:
            raise ValueError('opposite directions need different buttons')
        return [('button', negative), ('button', positive)]
    index = _index(spec['index'])
    if kind == 'trigger':
        rest, pressed = spec['rest'], spec['pressed']
        if not all(number(v) and -1 <= v <= 1 for v in (rest, pressed)) or rest == pressed:
            raise ValueError('trigger rest/pressed must be distinct values in [-1, 1]')
    return [('axis' if kind in ('axis', 'trigger') else 'button', index)]


def load_config(path):
    config = json.loads(Path(path).read_text())
    bindings = config['bindings']
    claimed = {('button', _index(bindings['quit_button'])): 'quit_button'}
    for name in ['servo{}'.format(joint) for joint in range(1, 6)] + ['gripper_open', 'gripper_close']:
        spec = bindings[name]
        for resource in binding_resources(spec, name.startswith('servo')):
            if resource in claimed:
                other_name = claimed[resource]
                other = bindings.get(other_name)
                # Some controllers put both triggers on one bipolar axis.
                shared_trigger = (name == 'gripper_close' and other_name == 'gripper_open'
                                  and spec['type'] == other['type'] == 'trigger'
                                  and spec['rest'] == other['rest'] == 0
                                  and spec['pressed'] == -other['pressed'])
                if not shared_trigger:
                    raise ValueError('input {} is shared by {} and {}'.format(resource, other_name, name))
            claimed[resource] = name
    if type(bindings.get('gripper_sign', 1)) is not int or bindings.get('gripper_sign', 1) not in (-1, 1):
        raise ValueError('gripper_sign must be -1 or 1')
    JogPlanner([sum(pair)/2 for pair in config['limits']], config['limits'],
               config['speed_deg_s'], config['deadzone'])
    TrackingMonitor(config.get('tracking_error_deg', 25), config.get('tracking_error_checks', 5))
    return config


def validate_device(bindings, axes, buttons):
    for name in ['servo{}'.format(joint) for joint in range(1, 6)] + ['gripper_open', 'gripper_close']:
        for kind, index in binding_resources(bindings[name], name.startswith('servo')):
            if index >= (len(axes) if kind == 'axis' else len(buttons)):
                raise ValueError('{} uses unavailable {} {}; use --inspect and edit the config'
                                 .format(name, kind, index))
    if bindings['quit_button'] >= len(buttons):
        raise ValueError('quit_button is unavailable; use --inspect and edit the config')


def binding_value(spec, axes, buttons):
    kind = spec['type']
    if kind == 'axis':
        return axes[spec['index']] * spec.get('sign', 1)
    if kind == 'buttons':
        return (int(buttons[spec['positive']])-int(buttons[spec['negative']])) * spec.get('sign', 1)
    if kind == 'button':
        return float(buttons[spec['index']])
    # Normalize triggers separately: an untouched -1 trigger is zero input.
    return max(0., min(1., (axes[spec['index']]-spec['rest']) / (spec['pressed']-spec['rest'])))


def servo_channels(bindings, axes, buttons, deadzone):
    inputs = [binding_value(bindings['servo{}'.format(joint)], axes, buttons)
              for joint in range(1, 6)]
    opening = binding_value(bindings['gripper_open'], axes, buttons)
    closing = binding_value(bindings['gripper_close'], axes, buttons)
    neutral = [abs(value) <= deadzone for value in inputs]
    neutral.append(max(opening, closing) <= deadzone)
    inputs.append((closing-opening) * bindings.get('gripper_sign', 1))
    return inputs, neutral


def servo_inputs(bindings, axes, buttons, deadzone):
    inputs, neutral = servo_channels(bindings, axes, buttons, deadzone)
    return inputs, all(neutral)


class NeutralGate:
    """Require each servo's own control to rest once, independently."""
    def __init__(self):
        self.active = [False]*6

    def update(self, neutral):
        newly_active = [index for index in range(6) if neutral[index] and not self.active[index]]
        for index in newly_active:
            self.active[index] = True
        return newly_active

    def filter(self, inputs):
        return [value if enabled else 0. for value, enabled in zip(inputs,self.active)]


def waiting_message(bindings, axes, buttons, active):
    blocked = []
    for index, enabled in enumerate(active):
        if enabled:
            continue
        names = ['servo{}'.format(index+1)] if index < 5 else ['gripper_open','gripper_close']
        values = []
        for name in names:
            spec = bindings[name]
            raw = ', '.join('{} {}={:.2f}'.format(kind, number,
                           axes[number] if kind == 'axis' else float(buttons[number]))
                           for kind,number in binding_resources(spec,index < 5))
            values.append('{} ({})'.format(name,raw))
        blocked.append(' / '.join(values))
    return 'Waiting for neutral: ' + '; '.join(blocked)


class TrackingMonitor:
    """Stop only on sustained position error on joints actually commanded."""
    def __init__(self, tolerance=25, checks=5):
        if not number(tolerance) or not 0 < tolerance <= 90:
            raise ValueError('tracking_error_deg must be between 0 (exclusive) and 90')
        if type(checks) is not int or not 1 <= checks <= 20:
            raise ValueError('tracking_error_checks must be an integer from 1 to 20')
        self.tolerance, self.checks = tolerance, checks
        self.failures = [0]*6

    def sample(self, measured, target, commanded):
        warnings = []
        for index, (actual, goal) in enumerate(zip(measured, target)):
            if index + 1 not in commanded or abs(actual-goal) <= self.tolerance:
                self.failures[index] = 0
                continue
            self.failures[index] += 1
            details = 'joint {}: read {} degrees, target {} degrees'.format(index+1, actual, goal)
            if self.failures[index] >= self.checks:
                raise RuntimeError('Sustained tracking error on {}; check torque, load, and I2C'.format(details))
            if self.failures[index] == 1:
                warnings.append('Tracking lag on {}; waiting for further readings'.format(details))
        return warnings


class LinuxGamepad:
    def __init__(self, path):
        self.fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        try:
            axes, buttons = bytearray(1), bytearray(1)
            fcntl.ioctl(self.fd, 0x80016a11, axes, True)  # JSIOCGAXES
            fcntl.ioctl(self.fd, 0x80016a12, buttons, True)  # JSIOCGBUTTONS
            self.axes, self.buttons = [0.] * axes[0], [False] * buttons[0]
        except BaseException:
            os.close(self.fd)
            raise

    def poll(self):
        count = 0
        while count < 1024:
            try:
                data = os.read(self.fd, _EVENT.size)
            except BlockingIOError:
                return count
            if len(data) != _EVENT.size:
                raise ConnectionError('Gamepad disconnected or returned an incomplete event')
            _, value, kind, index = _EVENT.unpack(data)
            kind &= ~0x80  # Includes initial-state events from the driver.
            if kind == 2 and index < len(self.axes):
                self.axes[index] = max(-1., min(1., value/32767.))
            elif kind == 1 and index < len(self.buttons):
                self.buttons[index] = bool(value)
            count += 1
        raise RuntimeError('Gamepad event queue is not draining; stopping control')

    def close(self):
        os.close(self.fd)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='/dev/input/js0')
    parser.add_argument('--config', default='cri/gamepad.json')
    parser.add_argument('--trace', action='store_true', help='Print command attempts and raw input diagnostics')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--inspect', action='store_true', help='Print input states; never open arm hardware')
    modes.add_argument('--dry-run', action='store_true', help='Simulate from a 90-degree pose; no arm hardware')
    args = parser.parse_args(argv)
    gamepad, arm = None, None
    try:
        config = None if args.inspect else load_config(args.config)
        gamepad = LinuxGamepad(args.device)
        print('Gamepad: {} axes, {} buttons'.format(len(gamepad.axes), len(gamepad.buttons)))
        if args.inspect:
            print('Move each stick and press buttons to identify indices. Ctrl+C exits.')
            while True:
                if gamepad.poll():
                    print('axes:', [round(value, 2) for value in gamepad.axes], 'buttons:',
                          [index for index, pressed in enumerate(gamepad.buttons) if pressed], flush=True)
                time.sleep(0.05)
        binds = config['bindings']
        validate_device(binds, gamepad.axes, gamepad.buttons)
        if args.dry_run:
            pose = [90] * 6
            print('DRY RUN: simulated initial pose is 90 degrees on all six joints.')
        else:
            if __package__:
                from .movement import ArmController
            else:
                from movement import ArmController
            arm = ArmController()
            pose = arm.read_angles()
        planner = JogPlanner(pose, config['limits'], config['speed_deg_s'], config['deadzone'])
        print('Initial angles:', pose)
        for warning in planner.limit_warnings(pose):
            print(warning, flush=True)
        print('Center each control once to enable its servo. Button {} quits.'
              .format(binds['quit_button']))
        print('Servo 1: left X; 2: left Y; 3: D-pad Y; 4: right Y; 5: right X; 6: triggers.')
        print('Torque unchanged. Limits:', planner.limits)
        gate = NeutralGate()
        waiting = None
        last_trace = float('-inf')
        commanded = set()
        monitor = TrackingMonitor(config.get('tracking_error_deg', 25),
                                  config.get('tracking_error_checks', 5))
        sent = tuple(int(round(value)) for value in pose)
        previous = last_check = time.monotonic()
        while True:
            started = time.monotonic()
            gamepad.poll()  # Disconnect errors exit before any subsequent writes.
            if gamepad.buttons[binds['quit_button']]:
                print('Quit button {} pressed.'.format(binds['quit_button']), flush=True)
                break
            inputs, neutral = servo_channels(binds, gamepad.axes, gamepad.buttons, planner.deadzone)
            newly_active = gate.update(neutral)
            if newly_active:
                if arm is not None:
                    measured = arm.read_angles()
                    planner.check_pose(measured)
                else:
                    measured = planner.target
                sent = list(sent)
                for index in newly_active:
                    planner.target[index] = float(measured[index])
                    sent[index] = int(round(measured[index]))
                sent = tuple(sent)
                print('Ready servos:', [index+1 for index,enabled in enumerate(gate.active) if enabled], flush=True)
            blocked = tuple(index for index,enabled in enumerate(gate.active) if not enabled)
            if blocked and blocked != waiting:
                print(waiting_message(binds, gamepad.axes, gamepad.buttons, gate.active), flush=True)
                print('Check --inspect and gamepad.json for any input that stays active at rest.', flush=True)
            waiting = blocked
            if args.trace and started-last_trace >= 1:
                print('Raw axes:', [round(value,2) for value in gamepad.axes],
                      'buttons:', [index for index,value in enumerate(gamepad.buttons) if value],
                      'inputs:', [round(value,2) for value in inputs], flush=True)
                last_trace = started
            inputs = gate.filter(inputs)
            if arm is not None and started-last_check >= 0.5:
                measured = arm.read_angles()
                planner.check_pose(measured)
                for warning in monitor.sample(measured, sent, commanded):
                    print(warning, flush=True)
                last_check = time.monotonic()
            target = planner.step(inputs, dt=started-previous)
            if target != sent:
                if arm is None:
                    print('Target:', target, flush=True)
                else:
                    for joint, (old, new) in enumerate(zip(sent, target), 1):
                        if new != old:
                            if args.trace:
                                print('Command: servo {} -> {} degrees (100 ms)'.format(joint,new), flush=True)
                            arm.move_servo(joint, new, s_time=100, wait=False)
                            commanded.add(joint)
                sent = target
            previous = started
            time.sleep(max(0, 0.1-(time.monotonic()-started)))
    except KeyboardInterrupt:
        pass
    except (OSError, ValueError, RuntimeError, ImportError, KeyError, TypeError) as exc:
        print('Control stopped: {}'.format(exc), file=sys.stderr)
        return 1
    finally:
        if gamepad is not None:
            gamepad.close()
        if arm is not None:
            arm.close()
        print('Stopped. No torque-release command sent; the last short move may finish.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
