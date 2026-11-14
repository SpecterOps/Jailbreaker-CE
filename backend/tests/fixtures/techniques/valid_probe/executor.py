class SourceControlledExecutor:
    async def execute(self, request, context):
        raise RuntimeError("fixture executor must not execute during package loading")
