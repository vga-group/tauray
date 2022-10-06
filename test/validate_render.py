import argparse
import subprocess
import sys
import tempfile

def validate_render(executable, scene, renderer, width, height, reference, metric, tolerance):
    with tempfile.TemporaryDirectory(prefix="tauray-test") as tmpdir:
        args = [
            executable,
            '--renderer='+renderer,
            '--width='+str(width),
            '--height='+str(height),
            '--headless='+tmpdir+'/frame',
            scene,
        ]
        if renderer == 'dshgi':
            args.append('--warmup-frames=100')
            args.append('--indirect-clamping=10')

        render = subprocess.run(capture_output=True, encoding='utf-8', args = args)
        render_command = ' '.join(render.args)
        if render.returncode != 0:
            print(render_command)
            print('Tauray returned error '+str(render.returncode)+'\nstdout:\n'+render.stdout+'\nstderr:\n'+render.stderr)
            return render.returncode

        compare = subprocess.run(capture_output=True, encoding='utf-8', args = [
            'compare',
            '-quiet', # disable warnings
            '-metric', metric,
            tmpdir+'/frame.exr',
            reference,
            'null:' # discard difference image
        ])
        if compare.returncode > 1:
            print(render_command)
            print('Compare returned error '+str(compare.returncode)+'\nstdout:\n'+compare.stdout+'\nstderr:\n'+compare.stderr)
            return compare.returncode

        if float(str(compare.stderr).split()[0]) > tolerance:
            print(render_command)
            print('Difference ' + str(compare.stderr).split()[0] + ' exceeds tolerance ' + str(tolerance))
            return -1
        # TODO: check for NaN/INF
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--executable')
    parser.add_argument('--scene')
    parser.add_argument('--renderer')
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--reference')
    parser.add_argument('--metric', default="mse")
    parser.add_argument('--tolerance', type=float)
    args = parser.parse_args()

    ret = validate_render(
        args.executable,
        args.scene,
        args.renderer,
        args.width,
        args.height,
        args.reference,
        args.metric,
        args.tolerance
    )

    sys.exit(ret);
