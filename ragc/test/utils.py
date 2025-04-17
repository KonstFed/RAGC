import os
import json
import textwrap
from typing import List, Tuple, Dict, Any
from torch_geometric.data import Data

from ragc.graphs.common import NodeTypeNumeric

def load_tasks(path: str | os.PathLike) -> List[Dict[str, Any]]:
    tasks = []
    with open(path, "r") as f:
        for line in f:
            js = json.loads(line)
            tasks.append(js)

    return tasks


def extract_signature(completion_path: str | os.PathLike, signature_position: Tuple[int, int]) -> str:
    start, end = signature_position
    with open(completion_path, "r") as f:
        completion_lines = f.read().split("\n")[start - 1 : end]

    signature = "\n".join(textwrap.dedent(line) for line in completion_lines)
    return signature


def map_cross_file_dependency(dependency: str, project_path: str, graph: Data) -> int | None:
    """This is human labeled mapping from evocodebench mapping to our.

    Main difference in handling of __init__.py.
    """
    new_name_map = {x: i for i, x in enumerate(graph.name)}
    new_name_map.update({x.removesuffix(".__init__"): i for i, x in enumerate(graph.name) if x.endswith("__init__") and graph.type[i] == NodeTypeNumeric.FILE.value})
    match project_path:
        case "Internet/pyramid":
            dependency = "build.lib." + dependency
        case (
            "Internet/Jinja2"
            | "Scientific-Engineering/bentoml"
            | "System/exodus-bundler"
            | "Text-Processing/mistune"
            | "Security/pyOpenSSL"
            | "Software-Development/pandas-profiling"
            | "System/viztracer"
        ):
            dependency = "src." + dependency


    corner_case = {
        "falcon.vendor.mimeparse.best_match":  "falcon.vendor.mimeparse.mimeparse.best_match",
        "falcon.media.json.JSONHandler.serialize": "falcon.media.json.JSONHandler._serialize_s",
        "chatette.cli.interactive_commands.command_strategy.CommandStrategy.command_tokens": "chatette.cli.interactive_commands.command_strategy.CommandStrategy",
        "chatette.cli.interactive_commands.command_strategy.CommandStrategy.print_wrapper": "chatette.cli.interactive_commands.command_strategy.CommandStrategy",
        "chatette.cli.interactive_commands.hide_command.HideCommand.stored_units": "chatette.cli.interactive_commands.hide_command.HideCommand",
        "mopidy.ext.Extension.ext_name": "mopidy.ext.Extension",
        "kinto.core.events.ACTIONS.CREATE": "kinto.core.events.ACTIONS",
        "kinto.core.events.ACTIONS.READ": "kinto.core.events.ACTIONS",
        "jc.jc_types.JSONDictType": "jc.jc_types",
        "alembic.util.rev_id": "alembic.util.langhelpers.rev_id",
        "build.lib.pyramid.interfaces.IRoute.pregenerator": "build.lib.pyramid.interfaces.IRoute",
        "bplustree.node.Node.page": "bplustree.node.Node",
        "alembic.operations.ops.AddColumnOp.column": "alembic.operations.ops.AddColumnOp",
        "alembic.langhelpers.Dispatcher.dispatch": "alembic.util.langhelpers.Dispatcher.dispatch",
        "recipe_ctx.RecipeCtx.ctx.ndk_api": "tests.recipes.recipe_ctx.RecipeCtx",
        "mopidy.config": "mopidy.config.__init__",
        "datasette.utils.escape_sqlite": "datasette.utils.__init__.escape_sqlite",
        "falcon.vendor.mimeparse": "falcon.vendor.mimeparse.mimeparse",
        "datasette.utils.path_with_added_args": "datasette.utils.__init__.path_with_added_args",
        "datasette.utils.path_with_removed_args": "datasette.utils.__init__.path_with_removed_args",
        "twilio.twiml.TwiML.nest": "twilio.twiml.__init__.TwiML.nest",
        "twilio.jwt.taskrouter.TaskRouterCapabilityToken._make_policy":"twilio.jwt.taskrouter.__init__.TaskRouterCapabilityToken._make_policy",
        "faker.providers.BaseProvider.numerify": "faker.providers.__init__.BaseProvider.numerify",
    }

    if dependency in corner_case:
        dependency = corner_case.get(dependency, dependency)
        return new_name_map[dependency]


    if 'alembic.script.ScriptDirectory' in dependency:
        dependency = dependency.replace('alembic.script.ScriptDirectory', 'alembic.script.base.ScriptDirectory')

    if 'alembic.environment' in dependency:
        dependency = dependency.replace('alembic.environment.', 'alembic.runtime.environment.')

    if dependency.startswith('ydata_profiling'):
        dependency = "src.ydata_profiling." + dependency.removeprefix("ydata_profiling.")

    if "mopidy.ext.ExtensionData.extension" in dependency:
        dependency = dependency.replace("mopidy.ext.ExtensionData.extension.", "mopidy.ext.Extension.")

    if dependency in [
            "src.jinja2.nodes.Node.environment",
            "falcon.constants.MEDIA_JSON",
            "boto.dynamodb2.types.QUERY_OPERATORS",
            "src.bentoml._internal.external_typing.NpNDArray",
            "pythonforandroid.logger.info",
            'falcon.app.App._error_handlers',
            'mrjob.step._JOB_STEP_FUNC_PARAMS',
            'twilio.base.values.unset',
            'kinto.core.logger',
            'mrjob.py2.string_types',
            'rest_framework.ISO_8601',
            'rest_framework.settings.api_settings',
            "alembic.util.not_none.add_nextrev"
            "falcon.vendor.mimeparse",
            "alembic.util.not_none.add_nextrev",
            "lux",
        ]:
        return None

    if dependency not in new_name_map:
        dependency = ".".join(dependency.split(".")[:-1])

    return new_name_map[dependency]
