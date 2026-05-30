try:
    from .utils import check_duplicate_nodes, log, color_text
    duplicate_dirs = check_duplicate_nodes()
    if duplicate_dirs:
        warning_msg = f"WARNING:  Found {len(duplicate_dirs)} other WanVideoWrapper directories:\n"
        for dir_path in duplicate_dirs:
            warning_msg += f"  - {color_text(dir_path, 'yellow')}\n"
        log.warning(color_text(warning_msg + "Please remove duplicates to avoid possible conflicts.", "red"))
except Exception:
    pass

from .utils import log

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Required modules (will raise on import failure)
REQUIRED_MODULES = [
    (".nodes", "Main"),
    (".nodes_sampler", "Sampler"),
    (".nodes_model_loading", "ModelLoading"),
    (".nodes_utility", "Utility"),
    (".cache_methods.nodes_cache", "Cache"),
]

# Optional modules (will warn on import failure)
OPTIONAL_MODULES = [
    (".nodes_deprecated", "Deprecated"),
    (".s2v.nodes", "S2V"),
    (".FlashVSR.flashvsr_nodes", "FlashVSR"),
    (".mocha.nodes", "Mocha"),
    (".fun_camera.nodes", "FunCamera"),
    (".uni3c.nodes", "Uni3C"),
    (".controlnet.nodes", "ControlNet"),
    (".ATI.nodes", "ATI"),
    (".multitalk.nodes", "MultiTalk"),
    (".recammaster.nodes", "RecamMaster"),
    (".skyreels.nodes", "SkyReels"),
    (".fantasytalking.nodes", "FantasyTalking"),
    (".qwen.qwen", "Qwen"),
    (".fantasyportrait.nodes", "FantasyPortrait"),
    (".unianimate.nodes", "UniAnimate"),
    (".MTV.nodes", "MTV"),
    (".HuMo.nodes", "HuMo"),
    (".lynx.nodes", "Lynx"),
    (".Ovi.nodes_ovi", "Ovi"),
    (".steadydancer.nodes", "SteadyDancer"),
    (".onetoall.nodes", "OneToAll"),
    (".WanMove.nodes", "WanMove"),
    (".SCAIL.nodes", "SCAIL"),
    (".LongCat.nodes", "LongCat"),
    (".LongVie2.nodes", "LongVie2"),
]

def register_nodes(module_path: str, name: str, optional: bool) -> None:
    """Import and register nodes from a module."""
    try:
        import importlib
        module = importlib.import_module(module_path, package=__package__)
        NODE_CLASS_MAPPINGS.update(getattr(module, "NODE_CLASS_MAPPINGS", {}))
        NODE_DISPLAY_NAME_MAPPINGS.update(getattr(module, "NODE_DISPLAY_NAME_MAPPINGS", {}))
    except Exception as e:
        if optional:
            log.warning(f"WanVideoWrapper WARNING: {name} nodes not available: {e}")
        else:
            raise

# Register all node modules
for module_path, name in REQUIRED_MODULES:
    register_nodes(module_path, name, optional=False)

for module_path, name in OPTIONAL_MODULES:
    register_nodes(module_path, name, optional=True)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
