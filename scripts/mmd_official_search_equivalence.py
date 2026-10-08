"""Official CBS.plan with root-only hooks; no alternative search algorithm.

The original plan AST is retained, including its search loop and return.
Only root construction may be skipped for repair-only diagnosis, and a root
transform may run immediately before the original root cost/conflict setup.
All arms use this single function and the unmodified official CBS.expand.
"""
import ast
import copy
import hashlib
import inspect
import textwrap
import types

from mmd.planners.multi_agent.cbs import CBS


def build_plan():
    source = textwrap.dedent(inspect.getsource(CBS.plan))
    tree = ast.parse(source)
    fn = tree.body[0]
    # Locate the original root block and root initialization success guard.
    root_start = next(i for i, n in enumerate(fn.body)
                      if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name)
                      and t.id == 'root_creation_start_time' for t in n.targets))
    root_end = next(i for i in range(root_start, len(fn.body))
                    if isinstance(fn.body[i], ast.If)
                    and ast.unparse(fn.body[i].test) == 'success_status == TrialSuccessStatus.UNKNOWN')
    original_search = copy.deepcopy(fn.body[root_end + 1:])
    original_root_setup = copy.deepcopy(fn.body[root_end].body)
    root_code = copy.deepcopy(fn.body[root_start:root_end])
    injected = ast.parse('root_creation_start_time = time.time()\nroot = initial_root').body
    fn.body[root_start:root_end] = [ast.If(
        test=ast.Compare(left=ast.Name(id='initial_root', ctx=ast.Load()),
                         ops=[ast.Is()], comparators=[ast.Constant(None)]),
        body=root_code, orelse=injected)]
    setup = fn.body[root_start + 1]
    setup.body[0:0] = ast.parse('if root_transform is not None:\n    root = root_transform(self, root)').body
    fn.args.args.extend([ast.arg(arg='initial_root'), ast.arg(arg='root_transform')])
    fn.args.defaults.extend([ast.Constant(None), ast.Constant(None)])
    ast.fix_missing_locations(tree)
    assert ast.dump(ast.Module(body=original_search, type_ignores=[])) == ast.dump(
        ast.Module(body=fn.body[root_start + 2:], type_ignores=[]))
    assert ast.dump(ast.Module(body=original_root_setup, type_ignores=[])) == ast.dump(
        ast.Module(body=setup.body[1:], type_ignores=[]))
    namespace = dict(CBS.plan.__globals__)
    exec(compile(tree, __file__, 'exec'), namespace)
    return namespace['plan'], {
        'official_source_file': inspect.getsourcefile(CBS.plan),
        'official_plan_sha256': hashlib.sha256(source.encode()).hexdigest(),
        'root_hook_source': ast.unparse(tree),
        'search_and_return_ast_identical': True,
        'root_cost_conflict_open_setup_ast_identical': True,
        'expand': inspect.getsourcefile(CBS.expand) + ':CBS.expand',
        'focal_note': 'Released code sorts open_l by len(conflict_l); no separate focal queue or focal weight.',
    }


PLAN, PROVENANCE = build_plan()


def bind_official_plan(planner):
    planner.plan = types.MethodType(PLAN, planner)
    assert 'expand' not in planner.__dict__
    assert planner.expand.__func__ is CBS.expand
    return planner
