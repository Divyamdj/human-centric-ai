from django.http import HttpResponse
from django.template import loader


def index(request):
    template = loader.get_template("index.html")

    students = [
        {"name": "Anagha Murgesh Hiremath", "matriculation": "640949"},
        {"name": "Divyam Jain", "matriculation": "638041"},
        {"name": "Subhajit Nandi", "matriculation": "641300"},
        {"name": "Aritra Paul", "matriculation": "641652"},
    ]

    projects = [
        {"name": "Project 1", "url_name": "project1:index"},
        {"name": "Project 2", "url_name": "project2:index"},
        {"name": "Project 3", "url_name": "project3:index"},
        {"name": "Project 4", "url_name": "project4:index"},
        {"name": "Project 5", "url_name": "project5:index"},
    ]

    context = {
        "students": students,
        "projects": projects,
    }

    return HttpResponse(template.render(context, request))
