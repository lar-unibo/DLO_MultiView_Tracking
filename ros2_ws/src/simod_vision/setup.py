from setuptools import find_packages, setup

package_name = "simod_vision"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="lar",
    maintainer_email="michela._cavuoto@libero.it",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "testing_dlo_proj = simod_vision.testing_dlo_proj:main",
            "feedback_camera = simod_vision.feedback_camera:main",
            "work_on_cable = simod_vision.work_on_cable:main",
            "init_dlo_shape = simod_vision.init_dlo_shape:main",
        ],
    },
)
