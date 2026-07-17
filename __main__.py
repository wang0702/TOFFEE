
import faulthandler
import signal
faulthandler.enable()
faulthandler.register(signal.SIGUSR1)

from toffee.cli import main

if __name__ == "__main__":
    main()
