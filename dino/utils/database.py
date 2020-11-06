from exlab.lab.database import Database
import pathlib

Database.set_databasedir(pathlib.Path(__file__).parent.parent.parent.absolute() / 'databases')